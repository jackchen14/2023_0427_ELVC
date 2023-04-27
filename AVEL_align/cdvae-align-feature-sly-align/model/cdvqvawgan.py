import os
import numpy as np
import math
import torch
import torch.nn.functional as F

from .radam import RAdam

from .layers import ( Conditions, log_loss, gradient_penalty_loss)
from .layers_vq import ( CDVectorQuantizer, EncodeResidualStack, DecodeResidualStack)
from .layers_pwg import MultiResolutionSTFTLoss

from .vqvawgan import Encoder, Decoder
from .vqvawgan import Model as VQVAE


class Trainer(object):
    def __init__(self, train_config, model_config):
        self.pre_iter      = train_config.get('pre_iter', 1000)
        self.iter_per_G    = train_config.get('iter_per_G', 1)
        self.iter_per_D    = train_config.get('iter_per_D', 1)
        self.iter_per_upd  = train_config.get('iter_per_upd', 8)
        self._gamma        = train_config.get('gamma', 1)
        self._gp_weight    = train_config.get('gp_weight', 1)
        self.gen_param     = train_config.get('generator_param', {
                                'optim_type': 'RAdam',
                                'learning_rate': 1e-4,
                                'max_grad_norm': 10,
                                'lr_scheduler':{
                                    'step_size': 100000,
                                    'gamma': 0.5,
                                    'last_epoch': -1
                                }
                            })
        self.disc_param    = train_config.get("discriminator_param", {
                                'optim_type': 'RAdam',
                                'learning_rate': 5e-5,
                                'max_grad_norm': 1,
                                'lr_scheduler':{
                                    'step_size': 100000,
                                    'gamma': 0.5,
                                    'last_epoch': -1
                                }
                            })

        self.stft_params   = train_config.get('stft_params', {'fft_sizes': [1024, 2048, 512],   # List of FFT size for STFT-based loss.
                                                              'hop_sizes': [120, 240, 50],      # List of hop size for STFT-based loss
                                                              'win_lengths': [600, 1200, 240],  # List of window length for STFT-based loss.
                                                              'window': 'hann_window'})              

        checkpoint_path    = train_config.get('checkpoint_path', '')


        # Initial Generator and Discriminator
        self.model_G = Model(model_config['Generator'])
        self.model_D = Encoder(**model_config['Discriminator'])

        print(self.model_G)
        print(self.model_D)

        self.mr_stft_loss = MultiResolutionSTFTLoss( fft_sizes=self.stft_params['fft_sizes'],
                                                     hop_sizes=self.stft_params['hop_sizes'],
                                                     win_lengths=self.stft_params['win_lengths'],
                                                     window=self.stft_params['window'])

        self.downsample_scale = np.prod(model_config['Discriminator']['downsample_scales'])

        # Initial Optimizer
        self.optimizer_G = RAdam( self.model_G.parameters(), 
                                  lr=self.gen_param['learning_rate'],
                                  betas=(0.5,0.999),
                                  weight_decay=0.0)

        self.optimizer_D = RAdam( self.model_D.parameters(), 
                                  lr=self.disc_param['learning_rate'],
                                  betas=(0.5,0.999),
                                  weight_decay=0.0)

        self.scheduler_G = torch.optim.lr_scheduler.StepLR(
                                    optimizer=self.optimizer_G,
                                    **self.gen_param['lr_scheduler']
                            )
        self.scheduler_D = torch.optim.lr_scheduler.StepLR(
                                    optimizer=self.optimizer_D,
                                    **self.disc_param['lr_scheduler']
                            )

        if os.path.exists(checkpoint_path):
            self.iteration = self.load_checkpoint(checkpoint_path)
        else:
            self.iteration = 0

        self.model_G.cuda().train()
        self.model_D.cuda().train()

    def step(self, input, iteration=None):
        if iteration is None:
            iteration = self.iteration

        assert self.model_G.training 
        assert self.model_D.training

        x_batch, y_batch = input
        x_batch, y_batch = [x.cuda() for x in x_batch], y_batch.cuda()

        loss_detail = dict() 

        ##########################
        # Phase 1: Train the VAE #
        ##########################
        if iteration <= self.pre_iter:
            x_real, x_fake, y_idx, loss, loss_detail = self.model_G((x_batch, y_batch), wav_loss_fn=self.mr_stft_loss)

            self.model_G.zero_grad()
            loss.backward()
            if self.gen_param['max_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model_G.parameters(),
                    self.gen_param['max_grad_norm'])
            self.optimizer_G.step()
            self.scheduler_G.step()

        ####################################
        # Phase 2: Train the whole network #
        ####################################
        elif iteration > self.pre_iter and iteration % self.iter_per_G == 0:
            # Train the Generator
            x_real, x_fake, y_idx, loss, loss_detail = self.model_G((x_batch, y_batch), wav_loss_fn=self.mr_stft_loss)
            y_idx = y_idx.repeat(1,int(x_fake.size(-1)/self.downsample_scale))
            adv_loss = F.nll_loss(self.model_D(x_fake), y_idx)            

            loss += self._gamma * adv_loss
            loss_detail['Total'] = loss.item()
            loss_detail['ADV loss'] = adv_loss.item()
            
            self.model_G.zero_grad()
            self.model_D.zero_grad()
            loss.backward()
            if self.gen_param['max_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model_G.parameters(),
                    self.gen_param['max_grad_norm'])
            self.optimizer_G.step()
            self.scheduler_G.step()

        ####################################
        # Phase 2: Train the discriminator #
        ####################################
        if iteration > self.pre_iter and iteration % self.iter_per_D == 0:
            # Train the Discriminator
            with torch.no_grad():
                x_real, x_fake, y_idx, _, _ = self.model_G((x_batch, y_batch), wav_loss_fn=self.mr_stft_loss)
            y_idx = y_idx.repeat(1,int(x_fake.size(-1)/self.downsample_scale))
            logit_real =  F.nll_loss(self.model_D(x_real), y_idx)
            logit_fake = -F.nll_loss(self.model_D(x_fake), y_idx)

            disc_loss = logit_real + logit_fake
            gp_loss = gradient_penalty_loss(x_real, x_fake, self.model_D)

            loss = disc_loss + self._gp_weight * gp_loss

            loss_detail['DISC loss'] = disc_loss.item()
            loss_detail['gradient_penalty'] = gp_loss.item()

            self.model_G.zero_grad()
            self.model_D.zero_grad()
            loss.backward()
            if self.disc_param['max_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model_D.parameters(),
                    self.disc_param['max_grad_norm'])
            self.optimizer_D.step()
            self.scheduler_D.step()


        if iteration is None:
            self.iteration += 1
        else:
            self.iteration = iteration

        return loss_detail


    def save_checkpoint(self, checkpoint_path):
        torch.save( {
                'model': self.model_G.state_dict(),
                'discriminator': self.model_D.state_dict(),
                'optimizer_G': self.optimizer_G.state_dict(),
                'optimizer_D': self.optimizer_D.state_dict(),
                'iteration': self.iteration,
            }, checkpoint_path)
        print("Saved state dict. to {}".format(checkpoint_path))


    def load_checkpoint(self, checkpoint_path):
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint_data.keys():
            self.model_G.load_state_dict(checkpoint_data['model'])
        # if 'discriminator' in checkpoint_data.keys():
        #     self.model_D.load_state_dict(checkpoint_data['discriminator'])
        if 'optimizer_G' in checkpoint_data.keys():
            self.optimizer_G.load_state_dict(checkpoint_data['optimizer_G'])
        # if 'optimizer_D' in checkpoint_data.keys():
        #     self.optimizer_D.load_state_dict(checkpoint_data['optimizer_D'])
        return checkpoint_data['iteration']

    def adjust_learning_rate(self, optimizer, learning_rate=None):
        if learning_rate is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate


class Model(torch.nn.Module):
    def __init__(self, arch):
        super(Model, self).__init__()

        self.encoder = torch.nn.ModuleDict()
        self.decoder = torch.nn.ModuleDict()

        self.encoder['mel'] = Encoder(**arch['mel']['encoder'])
        self.decoder['mel'] = Decoder(**arch['mel']['decoder'])
        self.encoder['wav'] = Encoder(**arch['wav']['encoder'])
        self.decoder['wav'] = Decoder(**arch['wav']['decoder'])

        self.quantizer = CDVectorQuantizer( arch['z_num'], arch['z_dim'], normalize=arch['embed_norm'], reduction='sum')
        self.embeds = Conditions( arch['y_num'], arch['y_dim'], normalize=True)
        
        self.beta = arch['beta']
        self.y_num = arch['y_num']  

        self.arch = arch


    def forward(self, input, encoder_kind='wav', decoder_kind='mel', wav_loss_fn=None):
        # Preprocess
        x, y_idx = input    # ( Size( N, x_dim, nframes), Size( N, nframes))
        y = self.embeds(y_idx).transpose(1,2).contiguous()    # Size( N, y_dim, nframes)

        if self.training:
            # Encode
            x_mel, x_wav = x

            z_mel = self.encoder['mel'](x_mel)
            z_wav = self.encoder['wav'](x_wav)

            # Decode
            z_vq, z_qut_loss, z_enc_loss, entropy = self.quantizer([z_mel,z_wav], ref='mean')
            z_mel_vq, z_wav_vq = z_vq

            xh_mel_mel = self.decoder['mel']((z_mel_vq, y))
            xh_wav_mel = self.decoder['mel']((z_wav_vq, y))
            xh_mel_wav = self.decoder['wav']((z_mel_vq, y))
            xh_wav_wav = self.decoder['wav']((z_wav_vq, y))

            # Loss
            batch_size, z_dim, frame_length = z_mel.shape
            mean_factor = batch_size

            z_qut_loss = z_qut_loss / mean_factor
            z_enc_loss = z_enc_loss / mean_factor / 2
            
            x_loss_mm = log_loss(xh_mel_mel, x_mel)
            x_loss_wm = log_loss(xh_wav_mel, x_mel)
            x_loss = (x_loss_mm + x_loss_wm) / mean_factor

            assert wav_loss_fn is not None
            sc_loss_mw, mag_loss_mw = wav_loss_fn( xh_mel_wav.squeeze(1), x_wav.squeeze(1))
            sc_loss_ww, mag_loss_ww = wav_loss_fn( xh_wav_wav.squeeze(1), x_wav.squeeze(1))
            sc_loss = sc_loss_mw + sc_loss_ww
            mag_loss = (mag_loss_mw + mag_loss_ww) / mean_factor

            loss = x_loss + sc_loss + mag_loss \
                    + z_qut_loss + self.beta * z_enc_loss
            
            losses = {'Total': loss.item(),
                      'VQ loss': z_enc_loss.item(),
                      'Entropy': entropy.item(),
                      'Mel loss': x_loss.item() / 2,
                      'Wav loss': mag_loss.item() / 2,
                      'SC loss': sc_loss.item() / 2,}

            y_idx = torch.zeros_like(y_idx,device=y_idx.device)
            return x_wav, xh_mel_wav, y_idx, loss, losses
            # return x, (xhat,xhat2), (y_idx,y2_idx), loss, losses

        else:
            z = self.encoder[encoder_kind](x)
            z_vq = self.quantizer([z], ref='0')
            xhat = self.decoder[decoder_kind]((z_vq, y))

            return xhat

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                # logging.debug(f"Weight norm is removed from {m}.")
                # print(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


    def get_marginal_model(self, encoder_kind, decoder_kind):
        arch_new = {'encoder':self.arch[encoder_kind]['encoder'],
                    'decoder':self.arch[decoder_kind]['decoder'],
                    'y_dim': self.arch['y_dim'],
                    'y_num': self.arch['y_num'],
                    'z_dim': self.arch['z_dim'],
                    'z_num': self.arch['z_num'],
                    'embed_norm': self.arch['embed_norm'],
                    'beta': self.arch['beta'],}

        model_new = VQVAE(arch_new)

        model_new.encoder.load_state_dict(self.encoder[encoder_kind].state_dict())
        model_new.decoder.load_state_dict(self.decoder[decoder_kind].state_dict())
        model_new.quantizer.load_state_dict(self.quantizer.state_dict())
        model_new.embeds.load_state_dict(self.embeds.state_dict())

        return model_new