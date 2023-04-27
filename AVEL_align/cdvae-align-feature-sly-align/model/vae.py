import torch
import torch.nn.functional as F

from .layers import ( Conv1d_Layernorm_LRelu, DeConv1d_Layernorm_GLU, 
                    GaussianSampler, GaussianKLD, GaussianLogDensity,
                    kl_loss, log_loss)


class Model(torch.nn.Module):
    def __init__(self, arch, normlize=False):
        super(Model, self).__init__()

        self.encoder = Encoder(arch['encoder'], arch['z_dim'])
        self.decoder = Decoder(arch['decoder'], arch['y_dim'])

        self.embeds = torch.nn.Embedding(   arch['y_num'], 
                                            arch['y_dim'], 
                                            padding_idx=None, 
                                            max_norm=None, 
                                            norm_type=2.0, 
                                            scale_grad_by_freq=False, 
                                            sparse=False, 
                                            _weight=None)
        self.normlize = normlize
        if self.normlize:
            self.embed_norm()

    def forward(self, input):
        # Preprocess
        if self.normlize:
            self.embed_norm()

        x, y = input    # ( Size( N, nframes, x_dim), Size( N, nframes))
        x = x.transpose(1,2).contiguous()    # Size( N, x_dim, nframes)
        y = self.embeds(y).transpose(1,2).contiguous()   # Size( N, y_dim, nframes)

        # Encode
        z_mu, z_lv = self.encoder(x)

        # Decode
        if self.training:
            z = GaussianSampler(z_mu, z_lv)
            xhat = self.decoder((z,y))
            
            # Loss functions
            batch_size = x.size(0)
            z_kld = kl_loss(z_mu, z_lv) / batch_size
            x_loss = log_loss(xhat, x) / batch_size

            sparsity = torch.mm(self.embeds.weight,self.embeds.weight.t())
            sparsity_target = torch.arange(sparsity.size(0),device=sparsity.device)
            sparsity = F.cross_entropy(sparsity,sparsity_target)

            loss = z_kld + x_loss + sparsity

            losses = {'Total': loss.item(),
                      'KLD': z_kld.item(),
                      'Sparsity': sparsity.item(),
                      'X like': x_loss.item()}

            return loss, losses

        else:
            xhat = self.decoder((z_mu,y))
            
            return xhat.transpose(1,2).contiguous()

    def embed_norm(self):
        with torch.no_grad():
            self.embeds.weight.div_(
                self.embeds.weight.norm(dim=1, keepdim=True)
            )


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
