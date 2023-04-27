import math

import torch
import torch.nn.functional as F

from .layers import ( Conditions, Conv1d_Layernorm_LRelu, DeConv1d_Layernorm_GLU,
                      log_loss)
from .layers_vq import CDVectorQuantizer, Jitter
import ipdb

class Model(torch.nn.Module):
    def __init__(self, arch):
        super(Model, self).__init__()

        self.encoder = torch.nn.ModuleDict()
        self.decoder = torch.nn.ModuleDict()

        self.encoder['sp'] = Encoder(arch['encoder']['sp'], arch['z_dim'])
        self.decoder['sp'] = Decoder(arch['decoder']['sp'], arch['y_dim'])

        self.encoder['mcc'] = Encoder(arch['encoder']['mcc'], arch['z_dim'])
        self.decoder['mcc'] = Decoder(arch['decoder']['mcc'], arch['y_dim'])

        self.quantizer = CDVectorQuantizer( arch['z_num'], arch['z_dim'], normalize=arch['embed_norm'], reduction='sum')
        self.spk_embeds = Conditions( arch['y_num'], arch['y_dim'], normalize=arch['embed_norm'])

        if arch['jitter_p'] > 0.0:
            self.jitter = Jitter(probability=arch['jitter_p'])
        else:
            self.jitter = None

        self.beta = arch['beta']
        self.arch = arch        

    def forward(self, input, encoder_kind='mcc', decoder_kind='mcc'):
        # Preprocess
        if self.training:
            #x_sp, x_mc, f0, y = input
            nl_sp ,nl_mcc ,y = input
            y = self.spk_embeds(y).transpose(1,2).contiguous()
            y = y.repeat(1,1,128)
            # ( Size( N, sp_dim, nframes), Size( N, mcc_dim, nframes), Size( N, nframes), Size( N, nframes))
        else:
            x, y = input
            y = self.spk_embeds(y).transpose(1,2).contiguous()
            y = y.repeat(1,1,x.size(-1))

        #y = self.spk_embeds(y).transpose(1,2).contiguous()   # Size( N, y_dim, nframes)
        ## repeat
        # print(y.shape)
        #y = y.repeat(1,1,128) ##(same same crop_length)
        # print(y.shape)
        # ipdb.set_trace()

        if self.training:
            # Encode
            z_sp = self.encoder['sp'](nl_sp)
            z_mc = self.encoder['mcc'](nl_mcc)

            # Vector Quantize
            z_vq, z_qut_loss, z_enc_loss, entropy = self.quantizer([z_sp,z_mc], ref='0')
            z_vq_sp, z_vq_mc = z_vq
            if self.jitter is not None:
                z_vq_sp, z_vq_mc = z_vq
                z_vq_sp = self.jitter(z_vq_sp)
                z_vq_mc = self.jitter(z_vq_mc)

            # Decode
            xh_sp_sp = self.decoder['sp']((z_vq_sp,y))
            xh_sp_mc = self.decoder['mcc']((z_vq_sp,y))
            xh_mc_sp = self.decoder['sp']((z_vq_mc,y))
            xh_mc_mc = self.decoder['mcc']((z_vq_mc,y))

            # Loss

            ## here maybe error
            batch_size = nl_sp.size(0)
            # print(nl_sp.size(0))
            # print(nl_sp.size(0))
            # ipdb.set_trace()

            z_qut_loss = z_qut_loss / batch_size
            z_enc_loss = z_enc_loss / batch_size

            sp_sp_loss = log_loss(xh_sp_sp, nl_sp) / batch_size
            sp_mc_loss = log_loss(xh_sp_mc, nl_mcc) / batch_size
            mc_sp_loss = log_loss(xh_mc_sp, nl_sp) / batch_size
            mc_mc_loss = log_loss(xh_mc_mc, nl_mcc) / batch_size

            loss = sp_sp_loss + sp_mc_loss + mc_sp_loss + mc_mc_loss + z_qut_loss + self.beta * z_enc_loss
            
            losses = {
                    'Total': loss.item(),
                    'VQ loss': z_qut_loss.item(),
                    'Entropy': entropy.item(),
                    'SP recon': sp_sp_loss.item(),
                    'SP cross': mc_sp_loss.item(),
                    'MCC recon': mc_mc_loss.item(),
                    'MCC cross': sp_mc_loss.item(),
                }

            return loss, losses

        else:
            # Encode
            z = self.encoder[encoder_kind](x)
            # Vector Quantize
            z = self.quantizer(z)
            # Decode
            xhat = self.decoder[encoder_kind]((z,y))

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