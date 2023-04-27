import math

import torch
import torch.nn.functional as F
from importlib import import_module

from .layers import ( Conditions, Conv1d_Layernorm_LRelu, DeConv1d_Layernorm_GLU,
                      log_loss)
from .layers_vq import CDVectorQuantizer, Jitter
import ipdb
import json
import copy

# Load Config.
str_config = "conf/config_stage1_mel_roy_cdvqvae_vc.json"
with open(str_config) as f:
    data = f.read()
config = json.loads(data)
model_config = config["model_config"]

### load org model ###
model_type = "stage1_mel_roy_cdvqvae"
org_module = import_module('model.{}'.format(model_type), package=None)
org_MODEL = getattr(org_module, 'Model')
org_model = org_MODEL(model_config)

model_path = "exp/checkpoints_stage1_mel_roy/06-25_01-00_600000"
org_model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
org_model.cuda()
org_decoder = copy.deepcopy(org_model.decoder['mel'])
org_encoder = copy.deepcopy(org_model.encoder['mel'])
Roy_encoder = copy.deepcopy(org_model.encoder['mel'])
for param in org_model.spk_embeds.parameters():
    param.requires_grad = False
org_spk_embeds = copy.deepcopy(org_model.spk_embeds)
print(org_decoder)
class Model(torch.nn.Module):
    def __init__(self, arch):
        super(Model, self).__init__()

        self.roy_encoder = torch.nn.ModuleDict()
        self.roy_encoder = Roy_encoder

        self.spk_embeds = org_spk_embeds
        for param in self.spk_embeds.parameters():
            param.requires_grad = False

        if arch['jitter_p'] > 0.0:
            self.jitter = Jitter(probability=arch['jitter_p'])
        else:
            self.jitter = None

        self.beta = arch['beta']
        self.arch = arch        

    def forward(self, input, encoder_kind='mel', decoder_kind='mel'):
        # Preprocess
        if self.training:
            #x_sp, x_mc, f0, y = input
            dtw_nl_sp ,dtw_nl_mel ,dtw_el_sp ,dtw_el_mel ,y = input
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
            z_roy_mel = self.roy_encoder(dtw_el_mel) ## roy latent
            with torch.no_grad():
                z_org_mel = org_encoder(dtw_nl_mel) ## org latent
            ## decode roy latent
            with torch.no_grad():
                xh_mel = org_decoder((z_roy_mel,y))

            batch_size = dtw_el_mel.size(0)

            L1_Loss = torch.nn.L1Loss(size_average= False)
            latent_loss = L1_Loss(z_roy_mel, z_org_mel) / batch_size
            mel_reconstruction_loss = log_loss(xh_mel, dtw_nl_mel) / batch_size

            loss = latent_loss + mel_reconstruction_loss
            
            losses = {
                    'Total': loss.item(),
                    'latent_loss': latent_loss.item(),
                    'mel_reconstruction_loss': mel_reconstruction_loss.item(),
                }

            # assert org_spk_embeds == org_model.spk_embeds
            for p1, p2 in zip(org_decoder.parameters(), org_model.decoder['mel'].parameters()):
                assert torch.equal(p1, p2)
            for p1, p2 in zip(org_encoder.parameters(), org_model.encoder['mel'].parameters()):
                assert torch.equal(p1, p2)
            # for param in self.spk_embeds.parameters():
            #     print("check")
            #     print(param)
            #     print(param.shape)
            # for param in org_model.spk_embeds.parameters():
            #     print("check")
            #     print(param)
            #     print(param.shape)
            # for (p1, p2) in (self.spk_embeds.parameters(), org_model.spk_embeds.parameters()):
 
                
            return loss, losses

        else:
            # Encode
            z = self.roy_encoder(x)
            # Decode
            xhat = org_decoder((z,y))

            return xhat

# class Encoder(torch.nn.Module):
#     def __init__(self, arch, z_dim):
#         super(Encoder, self).__init__()

#         self.layers = torch.nn.ModuleList()
#         for ( i, o, k, s) in zip( arch['input'], 
#                                   arch['output'], 
#                                   arch['kernel'], 
#                                   arch['stride']):
#             self.layers.append(
#                 Conv1d_Layernorm_LRelu( i, o, k, stride=s)
#             )

#         self.mlp = torch.nn.Conv1d( in_channels=arch['output'][-1],
#                                     out_channels=z_dim,
#                                     kernel_size=1)


#     def forward(self, input):
#         x = input   # Size( N, x_dim, nframes)
#         for i in range(len(self.layers)):
#             x = self.layers[i](x)
#         z = self.mlp(x)

#         return z   # Size( N, z_dim, nframes)

# class Decoder(torch.nn.Module):
#     def __init__(self, arch, y_dim):
#         super(Decoder, self).__init__()

#         self.layers = torch.nn.ModuleList()
#         for ( i, o, k, s) in zip( arch['input'], 
#                                   arch['output'], 
#                                   arch['kernel'], 
#                                   arch['stride']):
#             if len(self.layers) == len(arch['output']) - 1:
#                 self.layers.append(
#                     torch.nn.ConvTranspose1d( in_channels = i + y_dim,
#                                               out_channels=o,
#                                               kernel_size=k,
#                                               stride=s,
#                                               padding=int((k-1)/2))
#                 )                
#             else:
#                 self.layers.append(
#                     DeConv1d_Layernorm_GLU( i + y_dim, o, k, stride=s)
#                 )

#     def forward(self, input):
#         x, y = input   # ( Size( N, z_dim, nframes), Size( N, y_dim, nframes))

#         for i in range(len(self.layers)):
#             x = torch.cat((x,y),dim=1)
#             x = self.layers[i](x)

#         return x   # Size( N, x_dim, nframes)
