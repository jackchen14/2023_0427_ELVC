import math

import torch
import torch.nn.functional as F

from .radam import RAdam

from .layers import ( Conditions, Conv1d_Layernorm_LRelu, DeConv1d_Layernorm_GLU,
                      log_loss)
from .layers_vq import CDVectorQuantizer

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
            self.optimizer = RAdam( self.model.decoder.parameters(), 
                                    lr=self.opt_param['learning_rate'],
                                    betas=(0.5,0.999),
                                    weight_decay=0.0)
        else:
            self.optimizer = torch.optim.Adam( self.model.decoder.parameters(),
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
        if 'f0_embeds._embedding.weight' in checkpoint_data['model'].keys():
            state_dicts = checkpoint_data['model']
            state_dict_new = dict()
            for key in state_dicts.keys():
                model, feat = key.split('.')[:2]
                if model == 'encoder' and feat == 'mcc': 
                    key_new = '.'.join(key.split('.')[2:])
                    state_dict_new[key_new] = state_dicts[key]
            self.model.encoder.load_state_dict(state_dict_new)
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

        self.encoder = Encoder(arch['encoder']['mcc'], arch['z_dim'])
        self.decoder = Transformer(arch['decoder']['transformer'])

        self.quantizer = CDVectorQuantizer( arch['z_num'], arch['z_dim'], normalize=arch['embed_norm'], reduction='sum')
        self.spk_embeds = Conditions( arch['y_num'], arch['y_dim'], normalize=arch['embed_norm'])

        self.beta = arch['beta']
        self.arch = arch        

    def forward(self, input):
        # Preprocess
        if self.training:
            x_mc, x_mel, y, p_mc, p_mel = input
            # (Size( N, mcc_dim, nframes), Size( N, mel_dim, nframes), Size( N, nframes))
        else:
            x, y = input

        y = self.spk_embeds(y).transpose(1,2).contiguous()   # Size( N, y_dim, nframes)

        if self.training:
            with torch.no_grad():
                # Encode
                z_mc = self.encoder(x_mc)
                # Length normalize
                Batch, Dim, Time = z_mc.shape
                z_norm = z_mc.transpose(1,2).contiguous().view(-1, Dim)
                z_norm = z_norm / z_norm.norm(dim=1,keepdim=True)
                embedding = self.quantizer._embedding
                embedding = embedding / embedding.norm(dim=1, keepdim=True)
                # VQ
                distances = (torch.sum(z_norm.pow(2), dim=1, keepdim=True) 
                            + torch.sum(embedding.pow(2), dim=1)
                            - 2 * torch.matmul(z_norm, embedding.t()))
                z_id = torch.argmin(distances, dim=1)
                z_id = z_id.view(Batch, Time)

                # # Time Reduce
                z_id_tr = []
                p_mc_tr = []
                max_Time = 0

                for z_id_, p_mc_ in zip(z_id, p_mc):
                    z_id_ = torch.cat([z_id_,z_id_[-1:]*10000])
                    change = (z_id_[1:] - z_id_[:-1]).ne(0)
                    z_id_tr.append(z_id_[:-1].masked_select(change))
                    p_mc_ = torch.arange(1, p_mc_.masked_select(change).ne(0).sum()+1, device=p_mc_.device)
                    p_mc_[-1] *= -1
                    p_mc_tr.append(p_mc_)

                    # import ipdb
                    # ipdb.set_trace()

                    if z_id_tr[-1].size(0) > max_Time:
                        max_Time = z_id_tr[-1].size(0)

                z_id_tr = torch.cat([
                        F.pad(z_,(0,max_Time-z_.size(0)), 'constant').data.unsqueeze(0)
                        for z_ in z_id_tr
                    ], dim=0)

                p_mc_tr = torch.cat([
                        F.pad(p_,(0,max_Time-p_.size(0)), 'constant').data.unsqueeze(0)
                        for p_ in p_mc_tr
                    ], dim=0)

                z_vq_tr = embedding.index_select(dim=0,index=z_id_tr.view(-1)).view(Batch,max_Time,Dim)
                z_vq_tr = z_vq_tr.transpose(1,2)

            # Decode
            loss, losses = self.decoder((z_vq_tr,x_mel,y,p_mc_tr,p_mel))

            return loss, losses

        else:
            # Encode
            z = self.encoder(x)
            # Length normalize
            Batch, Dim, Time = z.shape
            z_norm = z.squeeze(0).t()
            z_norm = z_norm / z_norm.norm(dim=1,keepdim=True)
            embedding = self.quantizer._embedding
            embedding = embedding / embedding.norm(dim=1, keepdim=True)
            # VQ
            distances = (torch.sum(z_norm.pow(2), dim=1, keepdim=True) 
                        + torch.sum(embedding.pow(2), dim=1)
                        - 2 * torch.matmul(z_norm, embedding.t()))
            z_id = torch.argmin(distances, dim=1)
            # Time Reduce
            z_id_tr = torch.cat([z_id,z_id[-1:]*10000])
            change = (z_id_tr[1:] - z_id_tr[:-1]).ne(0)
            z_id_tr = z_id_tr[:-1].masked_select(change)
            z_vq_tr = embedding.index_select(dim=0,index=z_id_tr)
            z_vq_tr = z_vq_tr.t().unsqueeze(0)
            # Decode
            xhat = self.decoder((z_vq_tr,y))

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


