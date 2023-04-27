import torch
import torch.nn.functional as F

from .layers import ( Conv1d_Layernorm_LRelu, DeConv1d_Layernorm_GLU, 
                    GaussianSampler, GaussianKLD, GaussianLogDensity,
                    kl_loss, log_loss, skl_loss)

from .vae import VAE, Encoder, Decoder


class Model(torch.nn.Module):
    def __init__(self, arch):
        super(Model, self).__init__()

        self.encoder = torch.nn.ModuleDict()
        self.decoder = torch.nn.ModuleDict()

        self.encoder['sp'] = Encoder(arch['encoder']['sp'], arch['z_dim'])
        self.decoder['sp'] = Decoder(arch['decoder']['sp'], arch['y_dim'])

        self.encoder['mcc'] = Encoder(arch['encoder']['mcc'], arch['z_dim'])
        self.decoder['mcc'] = Decoder(arch['decoder']['mcc'], arch['y_dim'])

        self.embeds = torch.nn.Embedding(   arch['y_num'], 
                                            arch['y_dim'], 
                                            padding_idx=None, 
                                            max_norm=None, 
                                            norm_type=2.0, 
                                            scale_grad_by_freq=False, 
                                            sparse=False, 
                                            _weight=None)

        self.arch = arch

    def forward(self, input, encoder_kind='mcc', decoder_kind='mcc'):
        x, y = input    # ( Size( N, nframes, x_dim), Size( N, nframes))
        x = x.transpose(1,2).contiguous()    # Size( N, x_dim, nframes)
        y = self.embeds(y).transpose(1,2).contiguous()   # Size( N, y_dim, nframes)


        if self.training:
            x_sp, x_mc = torch.split( x, 513, 1) 
            # ( Size( N, sp_dim, nframes), Size( N, mcc_dim, nframes))

            z_sp_mu, z_sp_lv = self.encoder['sp'](x_sp)
            z_mc_mu, z_mc_lv = self.encoder['mcc'](x_mc)

            z_sp = GaussianSampler(z_sp_mu, z_sp_lv)
            z_mc = GaussianSampler(z_mc_mu, z_mc_lv)

            xh_sp_sp = self.decoder['sp']((z_sp,y))
            xh_sp_mc = self.decoder['mcc']((z_sp,y))
            xh_mc_sp = self.decoder['sp']((z_mc,y))
            xh_mc_mc = self.decoder['mcc']((z_mc,y))

            # Losses
            batch_size = x_sp.size(0)

            z_sp_kld = kl_loss(z_sp_mu, z_sp_lv) / batch_size
            z_mc_kld = kl_loss(z_mc_mu, z_mc_lv) / batch_size

            # z_loss =  skl_loss(z_sp_mu, z_sp_lv, z_mc_mu, z_mc_lv)
            # z_loss =  (z_sp - z_mc).pow(2).mean()
            z_loss =  (z_sp - z_mc).abs().mean()
            # z_loss = F.smooth_l1_loss( z_sp, z_mc)

            sp_sp_loss = log_loss(xh_sp_sp, x_sp) / batch_size
            sp_mc_loss = log_loss(xh_sp_mc, x_mc) / batch_size
            mc_sp_loss = log_loss(xh_mc_sp, x_sp) / batch_size
            mc_mc_loss = log_loss(xh_mc_mc, x_mc) / batch_size
            
            z_kld = z_sp_kld + z_mc_kld
            x_recon = sp_sp_loss + mc_mc_loss
            x_cross = sp_mc_loss + mc_sp_loss

            loss = z_kld + z_loss + x_recon + x_cross

            losses = {'Total': loss.item(),
                      'KLD': z_kld.item(),
                      'Z loss': z_loss.item(),
                      'X recon': x_recon.item(),
                      'X cross': x_cross.item()}

            return loss, losses

        else:
            z_mu, z_lv = self.encoder[encoder_kind](x)
            xhat = self.decoder[decoder_kind]((z_mu,y))

            return xhat.transpose(1,2).contiguous()



    def get_marginal_vae(self, encoder_kind, decoder_kind):
        arch_new = {'encoder':self.arch['encoder'][encoder_kind],
                    'decoder':self.arch['decoder'][decoder_kind],
                    'y_dim': 16,
                    'y_num': 12,
                    'z_dim': 16}
        vae_new = VAE(arch_new)

        vae_new.encoder.load_state_dict(self.encoder[encoder_kind].state_dict())
        vae_new.decoder.load_state_dict(self.decoder[decoder_kind].state_dict())
        vae_new.embeds.load_state_dict(self.embeds.state_dict())

        return vae_new

