import torch
import torch.nn.functional as F

from .layers import ( Conv1d_Layernorm_LRelu, DeConv1d_Layernorm_GLU, 
                      GaussianSampler, kl_loss, log_loss, 
                      gradient_penalty_loss)

from .vae import Encoder, Decoder

class Trainer(object):
    def __init__(self, train_config, model_config):
        self.learning_rate = train_config.get('learning_rate', 1e-4)
        self.iter_per_D    = train_config.get('iter_per_D', 5)
        self.pre_iter      = train_config.get('pre_iter', 1000)
        self.gan_iter      = train_config.get('gan_iter', 5000)
        self._gamma        = train_config.get('gamma', 50)
        self._gp_weight    = train_config.get('gp_weight', 10)

        self.model_G = Model(model_config['Generator'])
        self.model_D = Discriminator(model_config['Discriminator'])

        print(self.model_G)
        print(self.model_D)

        self.optimizer_G = torch.optim.Adam( self.model_G.parameters(), 
                                             lr=self.learning_rate,
                                             betas=(0.5,0.999),
                                             weight_decay=1e-6)

        self.optimizer_D = torch.optim.Adam( self.model_D.parameters(), 
                                             lr=self.learning_rate,
                                             betas=(0.5,0.999),
                                             weight_decay=1e-6)

        self.iteration = 0
        self.model_G.cuda().train()
        self.model_D.cuda().train()

    def step(self, input, iteration=None):
        if iteration is None:
            iteration = self.iteration

        assert self.model_G.training 
        assert self.model_D.training
        self.model_G.zero_grad()
        self.model_D.zero_grad()

        x_batch, y_batch = input
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        ##########################
        # Phase 1: Train the VAE #
        ##########################
        if iteration <= self.pre_iter:
            _, _, loss, loss_detail = self.model_G((x_batch, y_batch))
            loss.backward()
            self.optimizer_G.step()

        ####################################
        # Phase 2: Train the whole network #
        ####################################
        elif iteration > self.pre_iter and iteration % self.iter_per_D != 0:
            # Train the Discriminator
            x_real, x_fake, _, loss_detail = self.model_G((x_batch, y_batch))
            logit_real = self.model_D(x_real.detach())
            logit_fake = self.model_D(x_fake.detach())

            wgan_loss = logit_fake.mean() - logit_real.mean()
            gp_loss = gradient_penalty_loss(x_real, x_fake, self.model_D)

            loss = wgan_loss + self._gp_weight * gp_loss
            loss_detail['W-GAN loss'] = wgan_loss.item()
            loss_detail['gradient_penalty'] = gp_loss.item()
            
            loss.backward()
            self.optimizer_D.step()
        else:
            # Train the Generator
            x_real, x_fake, loss, loss_detail = self.model_G((x_batch, y_batch))
            logit_real = self.model_D(x_real)
            logit_fake = self.model_D(x_fake)

            wgan_loss = logit_fake.mean() - logit_real.mean()
            loss += -self._gamma * wgan_loss
            loss_detail['W-GAN loss'] = wgan_loss.item()

            loss.backward()
            self.optimizer_G.step()

        if iteration is None:
            self.iteration += 1
        else:
            self.iteration = iteration

        return loss_detail


    def save_checkpoint(self, checkpoint_path):
        torch.save( {
                'model': self.model_G.state_dict(),
                'iteration': self.iteration,
                'learning_rate': self.learning_rate,
            }, checkpoint_path)
        print("Saved state dict. to {}".format(checkpoint_path))


    def load_checkpoint(self, checkpoint_path):
        checkpoint_data = torch.load(checkpoint_path)
        self.model_G.load_state_dict(checkpoint_data['model'])
        self.optimizer = torch.optim.Adam( self.model.parameters(), 
                                           lr=checkpoint_data['learning_rate'],
                                           betas=(0.5,0.999),
                                           weight_decay=1e-6)

        return checkpoint_data['iteration']

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

            loss = z_kld + x_loss

            losses = {'Total': loss.item(),
                      'KLD': z_kld.item(),
                      'X like': x_loss.item()}

            return x, xhat, loss, losses
        else:
            xhat = self.decoder((z_mu,y))

            return xhat.transpose(1,2).contiguous()

    def embed_norm(self):
        with torch.no_grad():
            self.embeds.weight.div_(
                self.embeds.weight.norm(dim=1, keepdim=True)
            )


class Discriminator(torch.nn.Module):
    def __init__(self, arch):
        super(Discriminator, self).__init__()

        self.layers = torch.nn.ModuleList()
        for ( i, o, k, s) in zip( arch['input'], 
                                  arch['output'], 
                                  arch['kernel'], 
                                  arch['stride']):
            self.layers.append(
                Conv1d_Layernorm_LRelu( i, o, k, stride=s)
            )

        self.logit = torch.nn.Conv1d( in_channels=arch['output'][-1],
                                      out_channels=1,
                                      kernel_size=1)

    def forward(self, input):
        x = input   # Size( N, x_dim, nframes)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        x = self.logit(x) # Size( N, 1, nframes)

        return x.mean(dim=-1)   # Size( N, 1)