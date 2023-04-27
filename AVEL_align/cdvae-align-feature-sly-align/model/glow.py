# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import copy
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from .layers_glow import Invertible1x1Conv, WN, fNN, Squeeze1d


class Model(torch.nn.Module):
    def __init__(self, arch):
        super(Model, self).__init__()

        self.n_flows = arch['n_flows']
        self.n_group = arch['n_group']
        self.n_early_every = arch['n_early_every']
        self.n_early_size = arch['n_early_size']
        self.y_num = arch['y_num']
        self.y_dim = arch['y_dim']
        self.z_num = arch['z_num']
        self.z_dim = arch['z_dim']

        self.final_1x1Conv = True

        assert(self.n_group % 2 == 0)
        assert(self.n_early_size % 2 == 0)

        self.beta1, self.beta2 = arch['beta']
        

        # Initial speaker codes
        self.spk_embeds = torch.nn.Embedding(   self.y_num, 
                                                self.y_dim, 
                                                padding_idx=None, 
                                                max_norm=None, 
                                                norm_type=2.0, 
                                                scale_grad_by_freq=False, 
                                                sparse=False, 
                                                _weight=None)

        self.vq_embeds = torch.nn.Parameter( torch.randn(self.z_num, self.z_dim, requires_grad=True))

        # Initial model
        self.WN = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()

        n_remaining_channels = self.n_group
        n_half = int(self.n_group/2)

        # Set up layers with the right sizes based on how many dimensions
        # have been output already

        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_remaining_channels = n_remaining_channels - self.n_early_size
                n_half = int(n_remaining_channels/2)
            self.convinv.append(Invertible1x1Conv(n_remaining_channels, initial_type="random"))
            self.WN.append(fNN(n_half, **arch['WN_config']))
        if self.final_1x1Conv:
            self.final = Invertible1x1Conv(n_remaining_channels, initial_type="random")
        self.n_remaining_channels = n_remaining_channels  # Useful during inference

    def forward(self, input):
        x, y_idx = input    # ( Size( N, nframes, x_dim), Size( N, nframes))
        x = x.transpose(1,2).contiguous()    # Size( N, x_dim, nframes)

        y_0 = self.spk_embeds(torch.zeros_like(y_idx,device=y_idx.device).long()).transpose(1,2).contiguous()   # Size( N, y_dim, nframes)
        y = self.spk_embeds(y_idx).transpose(1,2).contiguous()   # Size( N, y_dim, nframes)

        if self.training:

            z, log_s_total, log_det_W_total = self.encode(x)

            glow_loss = - log_s_total - log_det_W_total
            glow_loss = glow_loss / (z.size(0)*z.size(1)*z.size(2))

            pho_code = z[:,:self.z_dim,:]
            spk_code = z[:,-self.y_dim:,:]     
            # z[:,-self.y_dim:,:] = z[:,-self.y_dim:,:] - y

            ################################     VQ Loss    ##################################
            
            # z_vq, entropy, sparsity_vq = self._vq(z[:,:self.z_dim,:])
            # z[:,:self.z_dim,:] = z[:,:self.z_dim,:] - z_vq

            ##################################################################################

            z_loss = z.pow(2).mean()

            # xhat = self.inverse(z_vq + y)
            # recon_loss = (xhat - x).pow(2).mean()

            sparsity_spk = -1 * (self.spk_embeds.weight[:,None] - self.spk_embeds.weight).pow(2).mean(dim=-1)
            sparsity_target = torch.arange(sparsity_spk.size(0),device=sparsity_spk.device)
            sparsity_spk = F.cross_entropy(sparsity_spk,sparsity_target)

            ################################     IR Loss    ##################################

            spk_code = spk_code.transpose(1,2).contiguous().view(-1,self.y_dim)
            i_loss = -1 * (spk_code[:,None] - self.spk_embeds.weight).pow(2).mean(dim=-1)
            i_loss = F.cross_entropy(i_loss,y_idx.view(-1))

            pho_code = pho_code.transpose(1,2).contiguous().view(-1,self.z_dim)
            r_loss = -1 * (pho_code[:,None] - self.spk_embeds.weight).pow(2).mean(dim=-1)
            r_loss = F.log_softmax(r_loss,dim=-1).mean(dim=-1).mean()

            ##################################################################################


            # loss = glow_loss + self.beta1 * z_loss  + self.beta2 * sparsity_spk # + recon_loss
            loss = glow_loss + z_loss + self.beta1 * i_loss  + self.beta2 * r_loss

            losses = {
                    'Total': loss.item(),
                    'Glow loss': glow_loss.item(),
                    'Z loss': z_loss.item(),
                    # 'Recon. loss': recon_loss.item(),
                    'Sparsity of spk': sparsity_spk.item(),
                    # 'Sparsity of vq': sparsity_vq.item(),
                    # 'Entropy': entropy.item(),
                    'i loss': i_loss.item(),
                    'r loss': r_loss.item(),
            }  

            return loss, losses

        else:

            z, _, _ = self.encode(x)

            # z[:,-self.y_dim:,:] += -z[:,-self.y_dim:,:].mean(dim=-1,keepdim=True)
            # z[:,-self.y_dim:,:] += - y_0
            
            # z[:,:self.z_dim,:],_,_ = self._vq(z[:,:self.z_dim,:])

            # z[:,-self.y_dim:,:] += y
            z[:,-self.y_dim:,:] = y

            xhat = self.inverse(z)

            # import ipdb
            # ipdb.set_trace()

            return xhat.transpose(1,2).contiguous()

    def _vq(self, z):
        z_t = z.detach().transpose(1,2).contiguous()
        z_flat = z_t.view(-1, z_t.size(2))
        with torch.no_grad():
            self.vq_embeds.div_(self.vq_embeds.norm(dim=1, keepdim=True))
        embedding = self.vq_embeds / self.vq_embeds.norm(dim=1, keepdim=True)

        distances = (torch.sum(z_flat.pow(2), dim=1, keepdim=True) 
                    + torch.sum(embedding.pow(2), dim=1)
                    - 2 * torch.matmul(z_flat, embedding.t()))
        
        encoding_idx = torch.argmin(distances, dim=1)
        if not self.training:
            print(encoding_idx)
        z_vq = self.vq_embeds.index_select(dim=0, index=encoding_idx).view(z_t.shape)

        encodings = torch.zeros(encoding_idx.shape[0], embedding.shape[0], device=z.device)
        encodings.scatter_(1, encoding_idx.unsqueeze(1), 1)

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        sparsity = torch.mm(embedding,embedding.t())
        sparsity_target = torch.arange(sparsity.size(0),device=sparsity.device)
        sparsity = F.cross_entropy(sparsity,sparsity_target)

        return z_vq.transpose(1,2).contiguous(), perplexity, sparsity

    def speaker_embed(self, x):
        x = x.transpose(1,2).contiguous()    # Size( N, x_dim, nframes)
        z, _, _ = self.encode(x)
        return z[:,-self.y_dim:,:].mean(dim=-1)

    def encode(self, x):
        z = []

        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                z.append(x[:,:self.n_early_size,:])
                x = x[:,self.n_early_size:,:]

            x, log_det_W = self.convinv[k](x)

            n_half = int(x.size(1)/2)
            x_0 = x[:,:n_half,:]
            x_1 = x[:,n_half:,:]

            output = self.WN[k](x_0)
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            x_1 = torch.exp(log_s)*x_1 + b

            if k == 0:
                log_s_total = torch.sum(log_s)
                log_det_W_total = log_det_W
            else:
                log_s_total += torch.sum(log_s)
                log_det_W_total += log_det_W

            x = torch.cat([x_0, x_1],1)

        if self.final_1x1Conv:
            x, log_det_W = self.final(x)
            log_det_W_total += log_det_W

        z.append(x)

        return torch.cat(z,1), log_s_total, log_det_W_total

    def inverse(self, z):
        """
        z = latent_code:  batch x data_dim x time
        """

        x = torch.autograd.Variable(z[:,-self.n_remaining_channels:,:])
        idx_s, idx_e = -self.n_remaining_channels-self.n_early_size, -self.n_remaining_channels

        if self.final_1x1Conv:
            x = self.final(x, reverse=True)

        for k in reversed(range(self.n_flows)):
            n_half = int(x.size(1)/2)
            x_0 = x[:,:n_half,:]
            x_1 = x[:,n_half:,:]

            output = self.WN[k](x_0)
            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            x_1 = (x_1 - b)/torch.exp(s)
            x = torch.cat([x_0, x_1],1)

            x = self.convinv[k](x, reverse=True)

            if k % self.n_early_every == 0 and k > 0:
                x = torch.cat((z[:,idx_s:idx_e,:], x),1)
                idx_s, idx_e = idx_s-self.n_early_size, idx_e-self.n_early_size

        return x

    @staticmethod
    def remove_weightnorm(model):
        waveglow = model
        for WN in waveglow.WN:
            # WN.start = torch.nn.utils.remove_weight_norm(WN.start)
            WN.in_layers = remove(WN.in_layers)
            # WN.cond_layers = remove(WN.cond_layers)
            # WN.res_skip_layers = remove(WN.res_skip_layers)
        return waveglow


def remove(conv_list):
    new_conv_list = torch.nn.ModuleList()
    for old_conv in conv_list:
        old_conv = torch.nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(old_conv)
    return new_conv_list
