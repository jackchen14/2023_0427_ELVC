import math

import torch
import torch.nn.functional as F

from .radam import RAdam

from .layers import ( Conditions, Conv1d_Layernorm_LRelu, DeConv1d_Layernorm_GLU,
                      log_loss)
from .layers_vq import CDVectorQuantizer
from .layers_align import Align_loss

from .transformer import Model as Transformer


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

        self.el_encoder = Encoder(arch['encoder']['el_mcc'], arch['z_dim'])
        self.encoder = Encoder(arch['encoder']['mcc'], arch['z_num'])
        if arch.get('use_tfm_decoder', False):
            self.decoder = Transformer(arch['decoder']['transformer'])
        else:
            self.decoder = Decoder(arch['decoder']['mel'], arch['y_dim'])

        self.quantizer = CDVectorQuantizer( arch['z_num'], arch['z_dim'], normalize=arch['embed_norm'], reduction='sum')
        self.spk_embeds = Conditions( arch['y_num'], arch['y_dim'], normalize=arch['embed_norm'])

        self.beta = arch['beta']
        self.arch = arch        

    def forward(self, input, encoder_kind='mcc', decoder_kind='mcc'):
        # Preprocess
        if self.training:
            x_el, x_nl, y, p_el, p_nl = input
            # ( Size( N, mcc_dim, nframes), Size( N, mcc_dim, nframes), Size( N, nframes), Size( N, 1))
        else:
            x, y = input

        y = self.spk_embeds(y).transpose(1,2).contiguous()   # Size( N, y_dim, nframes)

        if self.training:
            with torch.no_grad():
                # Encode
                z_nl = self.encoder(x_nl)
                Batch, Dim, Time = z_nl.shape
                z_norm = z_nl.transpose(1,2).contiguous().view(-1, Dim)
                z_norm = z_norm / z_norm.norm(dim=1,keepdim=True)
                # VQ
                embedding = self.quantizer._embedding
                embedding = embedding / embedding.norm(dim=1, keepdim=True)
                distances = (torch.sum(z_norm.pow(2), dim=1, keepdim=True) 
                            + torch.sum(embedding.pow(2), dim=1)
                            - 2 * torch.matmul(z_norm, embedding.t()))
                # Quantize
                z_id_nl = torch.argmin(distances, dim=1)
                z_id_nl = z_id_nl.view(Batch, Time)

                # p_nl = p_nl.sum(dim=-1)

                # # Time Reduce
                z_id_nl_tr = []
                p_nl_tr = []
                max_length = 0

                for z_id_nl_, p_nl_ in zip(z_id_nl,p_nl):
                    z_id_nl_ = torch.cat([z_id_nl_,z_id_nl_[-1:]*10000])
                    change = (z_id_nl_[1:] - z_id_nl_[:-1]).ne(0)
                    z_id_nl_tr.append(z_id_nl_[:-1].masked_select(change))
                    p_nl_tr.append(p_nl_.masked_select(change))
                    if z_id_nl_tr[-1].size(0) > max_length:
                        max_length = z_id_nl_tr[-1].size(0)

                z_id_nl_tr = torch.cat([
                        F.pad(z_,(0,max_length-z_.size(0)), 'constant').data.unsqueeze(0)
                        for z_ in z_id_nl_tr
                    ], dim=0)

                p_nl_tr = torch.cat([
                        p_.sum().unsqueeze(0)
                        for p_ in p_nl_tr
                    ], dim=0)

            # EL Encode
            z_p_el = self.el_encoder(x_el)
            z_p_el = F.log_softmax(z_p_el,dim=1).permute(2,0,1)

            # Loss
            lat_loss = F.ctc_loss( 
                    z_p_el, z_id_nl_tr, 
                    p_el, p_nl_tr, 
                    blank=0, 
                    reduction='mean', 
                    zero_infinity=False
                )

            loss = lat_loss
            
            losses = {
                    'Total': loss.item(),
                    'Latent CTC loss': lat_loss.item(),
                }

            return loss, losses

        else:
            # Encode
            z = self.el_encoder(x)
            # VQ
            # z_vq = self.quantizer([z], ref='0')
            z_el_id = torch.argmax(z.squeeze(0).t(), dim=1)

            # current_token = z_el_id[0]
            # for i in range(1,z_el_id.size(0)):
            #     if z_el_id[i] == 0:
            #         z_el_id[i] = current_token
            #     else:
            #         current_token = z_el_id[i]

            z_el_id = z_el_id.masked_select(z_el_id.ne(0))

            # z_el_id = z_el_id.unsqueeze(1).repeat(1,2).view(-1)

            z_el_id = torch.cat([z_el_id,z_el_id[-1:]*10000])
            change = (z_el_id[1:] - z_el_id[:-1]).ne(0)
            z_el_id = z_el_id[:-1].masked_select(change)

            embedding = self.quantizer._embedding
            embedding = embedding / embedding.norm(dim=1, keepdim=True)            
            z_vq = embedding.index_select(dim=0, index=z_el_id)
            z_vq = z_vq.t().unsqueeze(0)

            # Decode
            y = y.repeat(1,1,z_vq.size(2))
            xhat = self.decoder((z_vq,y))

            return xhat


class Encoder(torch.nn.Module):
    def __init__(self, arch, z_dim):
        super(Encoder, self).__init__()

        self.layers = torch.nn.ModuleList()
        use_dilate = arch.get('dilate', False)
        dilate = 0
        for ( i, o, k, s) in zip(  arch['input'], 
                                   arch['output'], 
                                   arch['kernel'], 
                                   arch['stride']):
            self.layers.append(
                Conv1d_Layernorm_LRelu( i, o, k, stride=s, dilation=2**dilate)
            )
            dilate = dilate + 1 if use_dilate else dilate

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
    x_el, x_nl, _, y, p_el, p_nl, _ = zip(*data)
    p_el = torch.cat([p.unsqueeze(0) for p in p_el],dim=0)
    p_nl = torch.cat([p.unsqueeze(0) for p in p_nl],dim=0)
    y = torch.cat([y_.unsqueeze(0) for y_ in y],dim=0)
    p_el = p_el.ne(0).sum(dim=-1)
    max_length = p_el.max()
    x_el = torch.cat([x[...,:max_length].unsqueeze(0) for x in x_el],dim=0)
    p_nl = p_nl.ne(0)
    max_length = p_nl.sum(dim=-1).max()
    p_nl = p_nl[:,:max_length]
    x_nl = torch.cat([x[...,:max_length].unsqueeze(0) for x in x_nl],dim=0)
    return x_el, x_nl, y, p_el, p_nl
