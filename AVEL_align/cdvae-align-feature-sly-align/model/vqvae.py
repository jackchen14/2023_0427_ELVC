import math

import torch
import torch.nn.functional as F

from .layers import ( Conditions, Conv1d_Layernorm_LRelu, DeConv1d_Layernorm_GLU,
                    log_loss)


class Model(torch.nn.Module):
    def __init__(self, arch):
        super(Model, self).__init__()

        self.encoder = Encoder(arch['encoder'], arch['z_dim'])
        self.decoder = Decoder(arch['decoder'], arch['y_dim'])

        self.quantizer = VectorQuantizer( arch['z_num'], arch['z_dim'], normalize=arch['embed_norm'])
        self.embeds = Conditions( arch['y_num'], arch['y_dim'], normalize=True)
        
        self.beta = arch['beta']

    def forward(self, input):
        # Preprocess
        x, y = input    # ( Size( N, nframes, x_dim), Size( N, nframes))
        x = x.transpose(1,2).contiguous()    # Size( N, x_dim, nframes)
        y = self.embeds(y).transpose(1,2).contiguous()    # Size( N, y_dim, nframes)
        # Encode
        z = self.encoder(x)

        # Decode
        if self.training:
            z_vq, z_qut_loss, z_enc_loss, entropy, sparsity_vq = self.quantizer(z)
            xhat = self.decoder((z_vq, y))

            # Loss
            batch_size = x.size(0)

            z_qut_loss = z_qut_loss / batch_size
            z_enc_loss = z_enc_loss / batch_size

            # z_enc_regu = z.detach().norm(dim=1).mean()
            # z_qut_regu = z_vq.detach().norm(dim=1).mean()
            
            x_loss = log_loss(xhat, x) / batch_size

            sparsity_spk = self.embeds.sparsity()

            loss = x_loss + z_qut_loss + self.beta * z_enc_loss
            
            losses = {'Total': loss.item(),
                      'VQ loss': z_enc_loss.item(),
                      'Entropy': entropy.item(),
                      'Sparsity of Spk': sparsity_spk.item(),
                      'Sparsity of VQ': sparsity_vq.item(),
                      'X like': x_loss.item()}

            return loss, losses

        else:
            
            z_vq = self.quantizer(z)
            xhat = self.decoder((z_vq,y))
            return xhat.transpose(1,2).contiguous()


class VectorQuantizer(torch.nn.Module):
    def __init__(self, z_num, z_dim, normalize=False):
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

    def embed_norm(self):
        if self.target_norm:
            with torch.no_grad():
                self._embedding.mul_(
                    self.target_norm / self._embedding.norm(dim=1, keepdim=True)
                )

    def forward(self, z):
        # z = z.permute(0, 2, 3, 1).contiguous()
        z = z.transpose(1,2).contiguous()
        z_shape = z.shape
        
        # Flatten & normalize
        if self.target_norm:
            z_norm = self.target_norm * z / z.norm(dim=2, keepdim=True)
            self.embed_norm()
            embedding = self.target_norm * self._embedding / self._embedding.norm(dim=1, keepdim=True)
        else:
            z_norm = z
            embedding = self._embedding
            
        z_flat = z_norm.view(-1, z.size(2))

        # Calculate distances
        distances = (torch.sum(z_flat.pow(2), dim=1, keepdim=True) 
                    + torch.sum(embedding.pow(2), dim=1)
                    - 2 * torch.matmul(z_flat, embedding.t()))
            
        # Quantize and unflatten
        encoding_idx = torch.argmin(distances, dim=1)
        z_vq = embedding.index_select(dim=0, index=encoding_idx).view(z_shape)
        
        if self.training:
            encodings = torch.zeros(encoding_idx.shape[0], embedding.shape[0], device=z.device)
            encodings.scatter_(1, encoding_idx.unsqueeze(1), 1)

            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

            sparsity = torch.mm(embedding,embedding.t())
            sparsity_target = torch.arange(sparsity.size(0),device=sparsity.device)
            sparsity = F.cross_entropy(sparsity,sparsity_target)

            z_qut_loss =  F.mse_loss(z_vq, z_norm.detach(), reduction='sum')
            z_enc_loss =  F.mse_loss(z_vq.detach(), z_norm, reduction='sum')
            if self.target_norm:
                z_enc_loss += F.mse_loss(z_norm, z, reduction='sum')    # Normalization loss

            z_vq = z_norm + (z_vq-z_norm).detach()
            
            return z_vq.transpose(1,2).contiguous(), z_qut_loss, z_enc_loss, perplexity, sparsity

        else:
            return z_vq.transpose(1,2).contiguous()


    def extra_repr(self):
        s = '{z_num}, {z_dim}'
        if self.normalize is not False:
            s += ', normalize=True'
        return s.format(**self.__dict__)

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

