import math

import torch
import torch.nn.functional as F
from importlib import import_module
from .radam import RAdam

from .layers import ( Conditions, Conv1d_Layernorm_LRelu, DeConv1d_Layernorm_GLU,
                      log_loss, GaussianSampler)
from .layers_vq import CDVectorQuantizer, Jitter
import ipdb
import json
import copy

class Trainer(object):
    def __init__(self, train_config, model_config):
        learning_rate  = train_config.get('learning_rate', 1e-4)
        model_type     = train_config.get('model_type', 'stage1_mel_wavlm_roy_cdvqvae')
        self.opt_param = train_config.get('optimize_param', {
                                'optim_type': 'RAdam',
                                'learning_rate': 1e-4,
                                'max_grad_norm': 10,
                                'lr_scheduler':{
                                    'step_size': 100000,
                                    'gamma': 0.5,
                                    'last_epoch': -1
                                }
                            })        

        model = Model(model_config).cuda()

        print(model)

        self.model = model.cuda()
        self.learning_rate = learning_rate

        if self.opt_param['optim_type'].upper() == 'RADAM':
            self.optimizer = RAdam( self.model.parameters(),
                                    lr=self.opt_param['learning_rate'],
                                    betas=(0.5,0.999),
                                    weight_decay=0.0)
        else:
            self.optimizer = torch.optim.Adam( self.model.parameters(),
                                               lr=self.opt_param['learning_rate'],
                                               betas=(0.5,0.999),
                                               weight_decay=0.0)

        if 'lr_scheduler' in self.opt_param.keys():
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                                optimizer=self.optimizer,
                                **self.opt_param['lr_scheduler']
                            )
        else:
            self.scheduler = None


        self.iteration = 0
        self.model.train()

    def step(self, input, iteration=None):
        assert self.model.training
        self.model.zero_grad()

        input = (x.cuda() for x in input)
        loss, loss_detail = self.model(input)

        loss.backward()
        if self.opt_param['max_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.opt_param['max_grad_norm'])
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if iteration is None:
            self.iteration += 1
        else:
            self.iteration = iteration

        return loss_detail


    def save_checkpoint(self, checkpoint_path):
        torch.save( {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iteration': self.iteration,
            }, checkpoint_path)
        print("Saved state dict. to {}".format(checkpoint_path))


    def load_checkpoint(self, checkpoint_path):
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        if 'encoder_check.WavLM.layers.0.conv.weight' not in checkpoint_data['model'].keys():
            print("sadly")
            print("-------------")
            state_dicts = checkpoint_data['model']
            enc_state_dict_new = dict()
            dec_state_dict_new = dict()
            for key in state_dicts.keys():
                model = key.split('.')[0]
                print(model)
                if model == 'encoder':
                    key_new = '.'.join(key.split('.')[1:])
                    enc_state_dict_new[key_new] = state_dicts[key]
                elif model == 'decoder':
                    key_new = '.'.join(key.split('.')[1:])
                    dec_state_dict_new[key_new] = state_dicts[key]
            self.model.encoder.load_state_dict(enc_state_dict_new)
            self.model.decoder.load_state_dict(dec_state_dict_new)
            self.model.encoder_check = copy.deepcopy(self.model.encoder)
            self.model.decoder_check = copy.deepcopy(self.model.decoder)
            # self.model.encoder_el = copy.deepcopy(self.model.encoder)
            self.model.decoder_el = copy.deepcopy(self.model.decoder)

            # state_dict_new = {'_embedding':state_dicts['quantizer._embedding']}
            # self.model.quantizer.load_state_dict(state_dict_new)
            state_dict_new = {'_embedding.weight':state_dicts['spk_embeds._embedding.weight']}
            self.model.spk_embeds.load_state_dict(state_dict_new)
            return 0
        else:
            print("good shit")
            print("-------------")

            self.model.load_state_dict(checkpoint_data['model'])
            self.optimizer.load_state_dict(checkpoint_data['optimizer'])
            self.scheduler.last_epoch = checkpoint_data['iteration']
            return checkpoint_data['iteration']

class Model(torch.nn.Module):
    def __init__(self, arch):
        super(Model, self).__init__()

        self.encoder = torch.nn.ModuleDict()
        self.decoder = torch.nn.ModuleDict()

        self.encoder['WavLM'] = Encoder(arch['encoder']['WavLM'], arch['z_dim'])
        self.decoder['WavLM'] = Decoder(arch['decoder']['WavLM'], arch['y_dim'])

        self.encoder['mel'] = Encoder(arch['encoder']['mel'], arch['z_dim'])
        self.decoder['mel'] = Decoder(arch['decoder']['mel'], arch['y_dim'])


        self.encoder_check = torch.nn.ModuleDict()
        self.decoder_check = torch.nn.ModuleDict()
        self.encoder_check = copy.deepcopy(self.encoder)
        self.decoder_check = copy.deepcopy(self.decoder)

        self.encoder_el = torch.nn.ModuleDict()
        self.decoder_el = torch.nn.ModuleDict()
        self.encoder_el = copy.deepcopy(self.encoder)
        self.decoder_el = copy.deepcopy(self.decoder)

        for param in self.encoder.parameters():#nn.Module有成员函数parameters()
            param.requires_grad = False
        for param in self.decoder.parameters():#nn.Module有成员函数parameters()
            param.requires_grad = False

        for param in self.encoder_el.parameters():#nn.Module有成员函数parameters()
            param.requires_grad = True
        for param in self.decoder_el.parameters():#nn.Module有成员函数parameters()
            param.requires_grad = False

        # self.quantizer = CDVectorQuantizer( arch['z_num'], arch['z_dim'], normalize=arch['embed_norm'], reduction='sum')
        self.spk_embeds = Conditions( arch['y_num'], arch['y_dim'], normalize=arch['embed_norm'])

        if arch['jitter_p'] > 0.0:
            self.jitter = Jitter(probability=arch['jitter_p'])
        else:
            self.jitter = None

        self.beta = arch['beta']
        self.arch = arch        

    def forward(self, input, encoder_kind='WavLM', decoder_kind='WavLM'):
        # Preprocess
        if self.training:
            #x_sp, x_mc, f0, y = input
            dtw_nl_mel ,dtw_nl_sp ,dtw_el_mel ,dtw_el_sp ,y = input
            y = self.spk_embeds(y).transpose(1,2).contiguous()
            y = y.repeat(1,1,128)
            # ( Size( N, sp_dim, nframes), Size( N, mcc_dim, nframes), Size( N, nframes), Size( N, nframes))
        else:
            x, y = input
            y = self.spk_embeds(y).transpose(1,2).contiguous()
            y = y.repeat(1,1,x.size(-1))

        #y = self.spk_embeds(y).transpose(1,2).contiguous()   # Size( N, y_dim, nframes)

        if self.training:
            # Encode to latent
            z_mel_mu_el,  z_mel_lv_el= self.encoder_el['mel'](dtw_el_mel) ## roy latent
            z_sp_mu_el,  z_sp_lv_el= self.encoder_el['WavLM'](dtw_el_sp)
            # z_mel = GaussianSampler(z_roy_mel_mu, z_roy_mel_lv)
            with torch.no_grad():
                z_mel_mu_nl,  z_mel_lv_nl = self.encoder['mel'](dtw_nl_mel)
                z_sp_mu_nl,  z_sp_lv_nl = self.encoder['WavLM'](dtw_nl_sp)
            
            # Decode
            xh_sp_sp = self.decoder_el['WavLM']((z_sp_mu_el,y))
            xh_sp_mel = self.decoder_el['mel']((z_sp_mu_el,y))
            xh_mel_sp = self.decoder_el['WavLM']((z_mel_mu_el,y))
            xh_mel_mel = self.decoder_el['mel']((z_mel_mu_el,y))

            batch_size = dtw_el_mel.size(0)

            L1_Loss = torch.nn.L1Loss(size_average= False)
            # latent_loss = \
            # L1_Loss(z_mel_mu_el, z_mel_mu_nl) / batch_size + \
            # L1_Loss(z_mel_mu_el, z_sp_mu_nl) / batch_size + \
            # L1_Loss(z_sp_mu_el, z_mel_mu_nl) / batch_size + \
            # L1_Loss(z_sp_mu_el, z_sp_mu_nl) / batch_size 

            latent_loss_1 = L1_Loss(z_mel_mu_el, z_mel_mu_nl) / batch_size
            latent_loss_2 = L1_Loss(z_mel_mu_el, z_sp_mu_nl) / batch_size
            latent_loss_3 = L1_Loss(z_sp_mu_el, z_mel_mu_nl) / batch_size
            latent_loss_4 = L1_Loss(z_sp_mu_el, z_sp_mu_nl) / batch_size
            latent_loss = latent_loss_1 + latent_loss_2 + latent_loss_3 + latent_loss_4 


            ## up 4 lines of code (latent_loss definition) can be modified
            ## cross domain latent losses may should be removed

            sp_sp_loss = log_loss(xh_sp_sp, dtw_nl_sp) / batch_size
            sp_mel_loss = log_loss(xh_sp_mel, dtw_nl_mel) / batch_size
            mel_sp_loss = log_loss(xh_mel_sp, dtw_nl_sp) / batch_size
            mel_mel_loss = log_loss(xh_mel_mel, dtw_nl_mel) / batch_size

            loss = latent_loss + sp_sp_loss + sp_mel_loss + mel_sp_loss + mel_mel_loss
            
            losses = {
                    'Total': loss.item(),
                    'latent_loss_1': latent_loss_1.item(),
                    'latent_loss_2': latent_loss_2.item(),
                    'latent_loss_3': latent_loss_3.item(),
                    'latent_loss_4': latent_loss_4.item(),
                    'WavLM recon': sp_sp_loss.item(),
                    'WavLM cross': sp_mel_loss.item(),
                    'Mel recon': mel_mel_loss.item(),
                    'Mel cross': mel_sp_loss.item(),
            }

            # assert org_spk_embeds == org_model.spk_embeds
            for p1, p2 in zip(self.encoder.parameters(), self.encoder_check.parameters()):
                assert torch.equal(p1, p2)
            for p1, p2 in zip(self.decoder.parameters(), self.decoder_check.parameters()):
                assert torch.equal(p1, p2)
              
            return loss, losses

        else:
            # Encode
            z_mu, z_lv= self.encoder_el[encoder_kind](x)
            xhat = self.decoder_el[decoder_kind]((z_mu,y))
            # Vector Quantize
            # z = self.quantizer(z)
            # Decode
            # xhat = org_decoder((z_mu,y))

            return xhat


class Encoder(torch.nn.Module):
    def __init__(self, arch, z_dim):
        super(Encoder, self).__init__()

        self.layers = torch.nn.ModuleList()
        for ( i, o, k, s) in zip( arch['input'], 
                                  arch['output'], 
                                  arch['kernel'], 
                                  arch['stride']):
            self.layers.append(
                Conv1d_Layernorm_LRelu( i, o, k, stride=s)
            )
        self.mlp_mu = torch.nn.Conv1d( in_channels=arch['output'][-1],
                                       out_channels=z_dim,
                                       kernel_size=1)
        self.mlp_lv = torch.nn.Conv1d( in_channels=arch['output'][-1],
                                       out_channels=z_dim,
                                       kernel_size=1)

    def forward(self, input):
        x = input   # Size( N, x_dim, nframes)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        z_mu = self.mlp_mu(x)
        z_lv = self.mlp_lv(x)

        return z_mu, z_lv   # Size( N, z_dim, nframes)

class Decoder(torch.nn.Module):
    def __init__(self, arch, y_dim):
        super(Decoder, self).__init__()

        self.layers = torch.nn.ModuleList()
        for ( i, o, k, s) in zip( arch['input'], 
                                  arch['output'], 
                                  arch['kernel'], 
                                  arch['stride']):
            if len(self.layers) == len(arch['output']) - 1:
                self.layers.append(
                    torch.nn.ConvTranspose1d( in_channels=i + y_dim,
                                              out_channels=o,
                                              kernel_size=k,
                                              stride=s,
                                              padding=int((k-1)/2))
                )                
            else:
                self.layers.append(
                    DeConv1d_Layernorm_GLU( i+y_dim, o, k, stride=s)
                )

    def forward(self, input):
        x, y = input   # ( Size( N, z_dim, nframes), Size( N, y_dim, nframes))

        for i in range(len(self.layers)):
            x = torch.cat((x,y),dim=1)
            x = self.layers[i](x)

        return x   # Size( N, x_dim, nframes)

    def forward(self, input):
        x, y = input   # ( Size( N, z_dim, nframes), Size( N, y_dim, nframes))

        for i in range(len(self.layers)):
            x = torch.cat((x,y),dim=1)
            x = self.layers[i](x)

        return x   # Size( N, x_dim, nframes)
