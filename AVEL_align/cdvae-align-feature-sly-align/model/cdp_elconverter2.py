import math

import torch
import torch.nn.functional as F

from .radam import RAdam

from .layers import ( Conditions, Conv1d_Layernorm_LRelu, DeConv1d_Layernorm_GLU,
                      log_loss)
from .layers_vq import CDVectorQuantizer
from .layers_align import Align_loss


class Trainer(object):
    def __init__(self, train_config, model_config):
        learning_rate  = train_config.get('learning_rate', 1e-4)
        model_type     = train_config.get('model_type', 'cdp_meldecoder')
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
            self.optimizer = RAdam( self.model.el_encoder.parameters(), 
                                    lr=self.opt_param['learning_rate'],
                                    betas=(0.5,0.999),
                                    weight_decay=0.0)
        else:
            self.optimizer = torch.optim.Adam( self.model.el_encoder.parameters(),
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
        if 'el_encoder.layers.0.conv.weight' not in checkpoint_data['model'].keys():
            state_dicts = checkpoint_data['model']
            enc_state_dict_new = dict()
            dec_state_dict_new = dict()
            for key in state_dicts.keys():
                model = key.split('.')[0]
                if model == 'encoder':
                    key_new = '.'.join(key.split('.')[1:])
                    enc_state_dict_new[key_new] = state_dicts[key]
                elif model == 'decoder':
                    key_new = '.'.join(key.split('.')[1:])
                    dec_state_dict_new[key_new] = state_dicts[key]
            self.model.encoder.load_state_dict(enc_state_dict_new)
            self.model.decoder.load_state_dict(dec_state_dict_new)
            state_dict_new = {'_embedding':state_dicts['quantizer._embedding']}
            self.model.quantizer.load_state_dict(state_dict_new)
            state_dict_new = {'_embedding.weight':state_dicts['spk_embeds._embedding.weight']}
            self.model.spk_embeds.load_state_dict(state_dict_new)
            return 0
        else:
            self.model.load_state_dict(checkpoint_data['model'])
            self.optimizer.load_state_dict(checkpoint_data['optimizer'])
            self.scheduler.last_epoch = checkpoint_data['iteration']
            return checkpoint_data['iteration']


class Model(torch.nn.Module):
    def __init__(self, arch):
        super(Model, self).__init__()

        self.el_encoder = Encoder(arch['encoder']['mcc'], arch['z_dim'])
        self.encoder = Encoder(arch['encoder']['mcc'], arch['z_dim'])
        self.decoder = Decoder(arch['decoder']['mel'], arch['y_dim'])

        self.quantizer = CDVectorQuantizer( arch['z_num'], arch['z_dim'], normalize=arch['embed_norm'], reduction='sum')
        self.spk_embeds = Conditions( arch['y_num'], arch['y_dim'], normalize=arch['embed_norm'])

        self.beta = arch['beta']
        self.arch = arch        

    def forward(self, input, encoder_kind='mcc', decoder_kind='mcc'):
        # Preprocess
        if self.training:
            x_el, x_mel, y, m_el, m_nl = input
            # ( Size( N, mcc_dim, nframes), Size( N, mcc_dim, nframes), Size( N, nframes), Size( N, 1))
        else:
            x, y = input

        y = self.spk_embeds(y).transpose(1,2).contiguous()   # Size( N, y_dim, nframes)

        if self.training:

            # EL Encode
            z_el = self.el_encoder(x_el)

            # EL VQ
            z_vq, _, z_enc_loss, entropy = self.quantizer([z_el], ref='0')
            z_vq = z_vq[0].masked_fill(m_el.unsqueeze(1), 0.0)

            # EL Decode
            y = y.repeat(1,1,z_vq.size(2))
            xh_mel = self.decoder((z_vq,y))

            # Loss
            batch_size = m_el.size(0)
            m_el = m_el.unsqueeze(1)
            len_el = m_el.sum(dim=-1,keepdim=True)

            z_enc_loss = z_enc_loss.div(batch_size*len_el).masked_select(m_el).sum()

            mel_loss = Align_loss(
                    xh_mel.transpose(1,2), 
                    x_mel.transpose(1,2), 
                    m_el.sum(dim=-1), 
                    m_nl.sum(dim=-1)
                ) / batch_size

            loss = mel_loss + z_enc_loss
            
            losses = {
                    'Total': loss.item(),
                    'Entropy': entropy.item(),
                    'Latent loss': z_enc_loss.item(),
                    'Mel loss': mel_loss.item(),
                }

            return loss, losses

        else:
            # Encode
            z = self.el_encoder(x)
            # VQ
            z_vq = self.quantizer([z], ref='0')
            # Decode
            y = y.repeat(1,1,x.size(2))
            xhat = self.decoder((z_vq,y))

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

        self.mlp = torch.nn.Conv1d( in_channels=arch['output'][-1],
                                    out_channels=z_dim,
                                    kernel_size=1)


    def forward(self, input):
        x = input   # Size( N, x_dim, nframes)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        z = self.mlp(x)

        return z   # Size( N, z_dim, nframes)


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
                    torch.nn.ConvTranspose1d( in_channels=i+y_dim,
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


def collate(data):
    x_el, x_mel, y, p_el, p_nl = zip(*data)
    m_el = torch.cat([p.ne(0).unsqueeze(0) for p in p_el],dim=0)
    m_nl = torch.cat([p.ne(0).unsqueeze(0) for p in p_nl],dim=0)
    y = torch.cat([y_.unsqueeze(0) for y_ in y],dim=0)
    max_length = m_el.sum(dim=-1).max()
    m_el = m_el[:,:max_length]
    x_el = torch.cat([x[...,:max_length].unsqueeze(0) for x in x_el],dim=0)
    max_length = m_nl.sum(dim=-1).max()
    m_nl = m_nl[:,:max_length]
    x_mel = torch.cat([x[...,:max_length].unsqueeze(0) for x in x_mel],dim=0)
    return x_el, x_mel, y, m_el, m_nl