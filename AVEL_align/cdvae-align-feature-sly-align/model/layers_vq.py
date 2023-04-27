import torch
import torch.nn.functional as F

from torch.autograd import grad as torch_grad

import math


class VectorQuantizer(torch.nn.Module):
    def __init__(self, z_num, z_dim, normalize=False, reduction='mean'):
        super(VectorQuantizer, self).__init__()

        if normalize:
            norm_scale = 0.25
            self.target_norm = 1.0 # norm_scale * math.sqrt(z.size(2))
        else:
            self.target_norm = None

        self._embedding = torch.nn.Parameter( torch.randn(z_num, z_dim, requires_grad=True))

        self.embed_norm()

        self.z_num = z_num
        self.z_dim = z_dim
        self.normalize = normalize
        self.reduction = reduction

    def embed_norm(self):
        if self.target_norm:
            with torch.no_grad():
                self._embedding.mul_(
                    self.target_norm / self._embedding.norm(dim=1, keepdim=True)
                )

    def forward(self, z, mask=None, time_last=True, time_reduction=False):
        # Flatten
        if time_last:
            B,D,T = z.shape
            z = z.transpose(1,2).contiguous().view(-1, D)
        else:
            B,T,D = z.shape
            z = z.contiguous().view(-1, D)
        device = z.device

        # Normalize
        if self.target_norm:
            z_norm = self.target_norm * z / z.norm(dim=1, keepdim=True)
            self.embed_norm()
            embedding = self.target_norm * self._embedding / self._embedding.norm(dim=1, keepdim=True)
        else:
            z_norm = z
            embedding = self._embedding

        # Calculate distances
        distances = (torch.sum(z_norm.pow(2), dim=1, keepdim=True) 
                    + torch.sum(embedding.pow(2), dim=1)
                    - 2 * torch.matmul(z_norm, embedding.t()))
            
        # Quantize
        encoding_idx = torch.argmin(distances, dim=1)
        z_vq = embedding.index_select(dim=0, index=encoding_idx)

        if mask is not None:
            mask = mask.view(-1)
            encoding_idx = (encoding_idx*mask + mask.eq(0).float()*self.z_num).long()

        # Calculate losses
        if self.training:
            encodings = torch.zeros(encoding_idx.size(0), embedding.size(0)+1, device=z.device)
            encodings.scatter_(1, encoding_idx.unsqueeze(1), 1)

            mean_factor = mask.sum() if mask is not None else encodings.size(0)
            avg_probs = torch.sum(encodings[:,:-1], dim=0) / mean_factor
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

            z_qut_loss = F.mse_loss(z_vq, z_norm.detach(), reduction=self.reduction)
            z_enc_loss = F.mse_loss(z_vq.detach(), z_norm, reduction=self.reduction)
            if self.target_norm:
                z_enc_loss += F.mse_loss(z_norm, z, reduction=self.reduction)    # Normalization loss
            if self.reduction == 'none':
                z_qut_loss = z_qut_loss.view(B,T,D)
                z_enc_loss = z_enc_loss.view(B,T,D)
                if time_last:
                    z_qut_loss = z_qut_loss.transpose(1,2)
                    z_enc_loss = z_enc_loss.transpose(1,2)                    

            z_vq = z_norm + (z_vq-z_norm).detach()

        # Time reduction
        if time_reduction:
            z_vq_new = []
            mask_new = []
            Ts = []
            for i in range(B):
                encoding_batch = encoding_idx[i*T:(i+1)*T]
                encoding_batch = F.pad(encoding_batch,(0,1),value=self.z_num)
                select_batch = (encoding_batch[1:] - encoding_batch[:-1]).ne(0)

                encoding_batch += F.pad(select_batch,(1,0)).cumsum(dim=0) * 1000

                select_batch = encoding_batch[:-1].masked_select(select_batch)
                same_map = (select_batch[:,None] == encoding_batch[None,:-1]).float()
                same_map = same_map / same_map.sum(dim=-1,keepdim=True)
                z_vq_tmp = torch.mm(same_map,z_vq[i*T:(i+1)*T])
                mask_tmp = torch.ones_like(select_batch,dtype=torch.float,device=device)
                z_vq_new.append(z_vq_tmp)
                mask_new.append(mask_tmp)
                Ts.append(mask_tmp.size(0))
            T = max(Ts) + 1
            z_vq = torch.cat([F.pad(zz,(0,0,0,T-t)) for t, zz in zip(Ts,z_vq_new)],dim=0)
            mask = torch.cat([F.pad(mm,(0,T-t)) for t, mm in zip(Ts,mask_new)],dim=0)

        # Deflatten
        z_vq = z_vq.view(B, T, D)
        if time_last:
            z_vq = z_vq.transpose(1,2).contiguous()
        if mask is not None:
            mask = mask.view(B,T)

        # Output
        if self.training and mask is not None:
            return z_vq, mask, z_qut_loss, z_enc_loss, perplexity
        elif self.training:
            return z_vq, z_qut_loss, z_enc_loss, perplexity
        elif mask is not None:
            return z_vq, mask
        else:
            return z_vq


    def sparsity(self):
        sparsity = torch.mm(self._embedding,self._embedding.t())
        sparsity_target = torch.arange(sparsity.size(0),device=sparsity.device)
        sparsity = F.cross_entropy(sparsity,sparsity_target)
        return sparsity

    def extra_repr(self):
        s = '{z_num}, {z_dim}'
        if self.normalize is not False:
            s += ', normalize=True'
        return s.format(**self.__dict__)


class CDVectorQuantizer(torch.nn.Module):
    def __init__(self, z_num, z_dim, normalize=False, reduction='mean'):
        super(CDVectorQuantizer, self).__init__()

        if normalize:
            norm_scale = 0.25
            self.target_norm = 1.0 # norm_scale * math.sqrt(z.size(2))
        else:
            self.target_norm = None

        self._embedding = torch.nn.Parameter( torch.randn(z_num, z_dim, requires_grad=True))

        self.embed_norm()

        self.z_num = z_num
        self.z_dim = z_dim
        self.normalize = normalize
        self.reduction = reduction

    def embed_norm(self):
        if self.target_norm:
            with torch.no_grad():
                self._embedding.mul_(
                    self.target_norm / self._embedding.norm(dim=1, keepdim=True)
                )

    def forward(self, z, ref='mean', time_last=True):
        z_dim_id = 1 if time_last else 2
        if not isinstance(z,list):
            z = [z]
        
        # Flatten & normalize
        if self.target_norm:
            z_norm = [self.target_norm * zz / zz.norm(dim=z_dim_id, keepdim=True) for zz in z]
            self.embed_norm()
            embedding = self.target_norm * self._embedding / self._embedding.norm(dim=1, keepdim=True)
        else:
            z_norm = z
            embedding = self._embedding
        
        # Select vectors in codebook
        with torch.no_grad():
            if ref == 'mean':
                z_ref = torch.cat([zz.unsqueeze(0) for zz in z_norm]).mean(dim=0)
            else:
                z_ref = z_norm[int(ref)]
            if time_last:
                z_ref = z_ref.transpose(1,2).contiguous()
            z_shape = z_ref.shape
            z_flat = z_ref.view(-1, z_ref.size(2))

            # Calculate distances
            distances = (torch.sum(z_flat.pow(2), dim=1, keepdim=True) 
                        + torch.sum(embedding.pow(2), dim=1)
                        - 2 * torch.matmul(z_flat, embedding.t()))
                
            # Quantize and unflatten
            encoding_idx = torch.argmin(distances, dim=1)
            
            
        z_vq = embedding.index_select(dim=0, index=encoding_idx).view(z_shape)

        if time_last:
            z_ref = z_ref.transpose(1,2).contiguous()
            z_vq = z_vq.transpose(1,2).contiguous()
        
        if self.training:
            encodings = torch.zeros(encoding_idx.shape[0], embedding.shape[0], device=z_ref.device)
            encodings.scatter_(1, encoding_idx.unsqueeze(1), 1)

            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

            z_qut_loss =  F.mse_loss(z_vq, z_ref.detach(), reduction=self.reduction)
            z_enc_loss = 0.0
            for i,zz in enumerate(z_norm):
                z_enc_loss += F.mse_loss(z_vq.detach(), zz, reduction=self.reduction)
                if self.target_norm:
                    z_enc_loss += F.mse_loss(zz, z[i], reduction=self.reduction)    # Normalization loss

            z_vq = [(zz + (z_vq-zz).detach()) for zz in z_norm]

            return z_vq, z_qut_loss, z_enc_loss, perplexity

        else:
            return z_vq


    def extra_repr(self):
        s = '{z_num}, {z_dim}'
        if self.normalize is not False:
            s += ', normalize=True'
        return s.format(**self.__dict__)


class EncodeResidualStack(torch.nn.Module):
    def __init__(self,
                 kernel_size=3,
                 channels=128,
                 dilation=1,
                 bias=True,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 normalization_func="GroupNorm",
                 normalization_params={ "num_groups": 1,
                                        "eps": 1e-05, 
                                        "affine": True},                 
                 pad="ReflectionPad1d",
                 pad_params={},
                 use_causal_conv=False,
                 ):
        super(EncodeResidualStack, self).__init__()

        assert not use_causal_conv, "Not supported yet."
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        padding1 = (kernel_size - 1) // 2 * dilation
        padding2 = (kernel_size - 1) // 2
        normalization_params['num_channels'] = channels

        self.stack = torch.nn.Sequential(
            getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            getattr(torch.nn, pad)(padding1, **pad_params),
            torch.nn.Conv1d(channels, channels, kernel_size, dilation=dilation, bias=bias),
            getattr(torch.nn, normalization_func)(**normalization_params),
            getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            getattr(torch.nn, pad)(padding2, **pad_params),
            torch.nn.Conv1d(channels, channels, kernel_size, bias=bias),
            getattr(torch.nn, normalization_func)(**normalization_params),
        )

        self.skip_layer = torch.nn.Conv1d( channels, channels, 1, bias=bias)


    def forward(self, c):
        """Calculate forward propagation.
        Args:
            c (Tensor): Input tensor (B, channels, T).
        Returns:
            Tensor: Output tensor (B, chennels, T).
        """
        return self.stack(c) + self.skip_layer(c)


class DecodeResidualStack(torch.nn.Module):
    def __init__(self,
                 kernel_size=3,
                 in_channels=128,
                 cond_channels=128,
                 skip_channels=80,
                 dilation=0,
                 bias=True,
                 dropout=0.0,
                 nonlinear_activation="GLU",
                 nonlinear_activation_params={},
                 normalization_func="GroupNorm",
                 normalization_params={ "num_groups": 2,
                                        "num_channels": 256,
                                        "eps": 1e-05, 
                                        "affine": True},
                 pad="ReflectionPad1d",
                 pad_params={},             
                 use_causal_conv=False,
                 ):
        super(DecodeResidualStack, self).__init__()
        assert nonlinear_activation == "GLU", "Not supported yet."

        if not use_causal_conv:
            assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."

        self.use_causal_conv = use_causal_conv
        self.dropout = dropout

        padding = (kernel_size - 1) // 2 * dilation
        self.conv_in = torch.nn.ConvTranspose1d(in_channels, in_channels*2, kernel_size,
                            padding=padding, dilation=dilation, bias=bias)
        
        self.conv_cond = torch.nn.Conv1d(cond_channels, in_channels*2, 1, bias=bias)

        normalization_params["num_channels"] = in_channels*2
        self.norm_layer = getattr(torch.nn, normalization_func)(**normalization_params)

        self.res_skip_layers = torch.nn.Conv1d(in_channels, in_channels+skip_channels, 1)

        self.in_channels = in_channels


    def forward(self, x, c):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            c (Tensor): Local conditioning auxiliary tensor (B, cond_channels, T).
        Returns:
            Tensor: Output tensor for skip connection (B, in_channels, T).
        """

        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_res = self.conv_in(x) + self.conv_cond(c)
        x_res = self.norm_layer(x_res)
        x_res_tanh = torch.tanh(x_res[:,:self.in_channels])
        x_res_sigmoid = torch.sigmoid(x_res[:,self.in_channels:])
        x_res = x_res_tanh * x_res_sigmoid

        x_res_skip = self.res_skip_layers(x_res)
        
        x = x_res_skip[:,:self.in_channels,:] + x
        x_skip = x_res_skip[:,self.in_channels:,:]

        return x, x_skip


import torch.nn as nn
import numpy as np


class Jitter(nn.Module):
    """
    Jitter implementation from [Chorowski et al., 2019].
    During training, each latent vector can replace either one or both of
    its neighbors. As in dropout, this prevents the model from
    relying on consistency across groups of tokens. Additionally,
    this regularization also promotes latent representation stability
    over time: a latent vector extracted at time step t must strive
    to also be useful at time steps t − 1 or t + 1.
    """

    def __init__(self, probability=0.12):
        super(Jitter, self).__init__()

        self._probability = probability

    def forward(self, quantized):
        original_quantized = quantized.detach().clone()
        length = original_quantized.size(2)
        for i in range(length):
            """
            Each latent vector is replace with either of its neighbors with a certain probability
            (0.12 from the paper).
            """
            replace = [True, False][np.random.choice([1, 0], p=[self._probability, 1 - self._probability])]
            if replace:
                if i == 0:
                    neighbor_index = i + 1
                elif i == length - 1:
                    neighbor_index = i - 1
                else:
                    """
                    "We independently sample whether it is to
                    be replaced with the token right after
                    or before it."
                    """
                    neighbor_index = i + np.random.choice([-1, 1], p=[0.5, 0.5])
                quantized[:, :, i] = original_quantized[:, :, neighbor_index]

        return quantized

    def extra_repr(self):
        s = 'jitter_prob={_probability}'
        return s.format(**self.__dict__)

