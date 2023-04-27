import os
import copy

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from .radam import RAdam

from .layers import ( Conditions, gradient_penalty_loss_S2S)
from .layers_vq import VectorQuantizer, EncodeResidualStack
from .layers_tf import (clones, Linear, Conv, Attention, FFN, 
                        DecoderPrenet, get_sinusoid_encoding_table)


class Model(nn.Module):
    """
    Transformer Network
    """
    def __init__(self, arch):
        super(Model, self).__init__()
        self.encoder = Encoder(**arch['encoder'])
        self.decoder = Decoder(**arch['decoder'])

        if 'z_num' in arch.keys():
            self.word_embeddings = Conditions( arch['z_num'], arch['z_dim'], normalize=False)
        else:
            self.word_embeddings = None
        if 'y_num' in arch.keys():
            self.spk_embeddings = Conditions( arch['y_num'], arch['y_dim'], normalize=True)
        else:
            self.spk_embeddings = None
        
        self.gamma = arch.get('gamma', 8)


    def forward(self, input):
        if self.training:
            word_in, mel_in, spk, pos_word, pos_mel = input
            # word_in with Size( N, T1), or Size( N, z_dim, T1)
            # mel_in with Size( N, mel_dim, T2)
            # spk with Size( N, 1), or Size( N, y_dim, 1)
            # pos_word with Size( N, T1)
            # pos_mel with Size( N, T2)
            mel_in = mel_in.transpose(1,2).contiguous()
        else:
            word_in, spk = input
            # word_in with Size( N, T1), or Size( N, z_dim, T1)
            # spk with Size( N, 1), or Size( N, y_dim, 1)
            pos_word = t.arange(1,word_in.size(-1)+1,device=word_in.device).unsqueeze(0)

        if self.word_embeddings is not None and word_in.ndim == 2:
            word_in = self.word_embeddings(word_in)          # Size( N, T1, z_dim)
        else:
            word_in = word_in.transpose(1,2).contiguous()

        if self.spk_embeddings is not None and spk.ndim == 2:
            spk = self.spk_embeddings(spk_id[:,:1])          # Size( N, 1, y_dim)
        else:
            spk = spk[:,:,:1].transpose(1,2).contiguous()

        memory, c_mask = self.encoder(word_in, pos_word.abs())
        memory_spk = t.cat([memory,spk.repeat(1,memory.size(1),1)],dim=2)

        if self.training:            
            mel_input = t.cat([mel_in[:,:1,:]*0.0, mel_in[:,:-1,:]],dim=1).detach()
            mel_pred, post_pred, stop_preds = self.decoder(memory_spk, mel_input, c_mask, pos_mel.abs())       

            mask = pos_mel.ne(0).unsqueeze(-1)
            length = mask.float().sum(dim=1,keepdim=True)
            batch_size = word_in.size(0)
            mean_factor =  batch_size * length

            # Mel Prediction
            mel_loss = (mel_pred - mel_in).pow(2).div(mean_factor).masked_select(mask).sum()
            post_loss = (post_pred - mel_in).pow(2).div(mean_factor).masked_select(mask).sum()

            # Stop Prediction
            stop_tokens = pos_mel.lt(0).float().unsqueeze(-1)
            stop_weights = stop_tokens * (self.gamma - 1 ) + mask.float()

            stop_loss = F.binary_cross_entropy( stop_preds, stop_tokens, weight=stop_weights, reduction='none')
            stop_loss = stop_loss.div(mean_factor).masked_select(mask).sum()
            # stop_accuracy = (stop_preds.masked_select(stop_tokens.eq(1.0)).ge(0.5)).float().masked_select(mask).mean()
 
            loss = mel_loss + post_loss + stop_loss

            loss_detail = { 'Total': loss.item(),
                            'Mel loss': mel_loss.item(),
                            'Post loss': post_loss.item(),
                            'Stop loss': stop_loss.item()}
                            
            return loss, loss_detail

        else:
            device = memory_spk.device
            mel_input = t.zeros(1,1,80).float().to(device)

            MAX_LENGTH = 1000
            pos_mel = t.arange(1,MAX_LENGTH+1,device=device).unsqueeze(0)

            for i in range(1,MAX_LENGTH+1):

                _, post_pred, stop_preds = self.decoder( memory_spk, mel_input, c_mask, pos_mel[:,:i])

                if stop_preds[:,-1,0] > 0.5:
                    break
                # mel_input = t.cat([mel_input[:,:1] * 0.0, post_pred],dim=1)
                mel_input = t.cat([mel_input, post_pred[:,-1:]],dim=1)

            return post_pred.transpose(1,2).contiguous()


class EncoderPrenet(nn.Module):
    """
    Pre-network for Encoder consists of convolution networks.
    """
    def __init__(self, num_input, num_hidden, num_output, dropout_p=0.5, norm_func='batch_norm'):
        super(EncoderPrenet, self).__init__()

        self.conv1 = Conv(in_channels=num_input,
                          out_channels=num_hidden,
                          kernel_size=5,
                          dilation=1,
                          stride=1,
                          padding=2,
                          w_init='relu')
        self.conv2 = Conv(in_channels=num_hidden,
                          out_channels=num_hidden,
                          kernel_size=5,
                          dilation=1,
                          stride=2,
                          padding=2,
                          w_init='relu')
        self.conv3 = Conv(in_channels=num_hidden,
                          out_channels=num_hidden,
                          kernel_size=5,
                          dilation=1,
                          stride=2,
                          padding=2,
                          w_init='relu')

        if norm_func == 'batch_norm':
            self.norm1 = nn.BatchNorm1d(num_hidden)
            self.norm2 = nn.BatchNorm1d(num_hidden)
            self.norm3 = nn.BatchNorm1d(num_hidden)
        else:
            self.norm1 = nn.GroupNorm( num_groups=1, num_channels=num_hidden, eps=1e-5, affine=True)
            self.norm2 = nn.GroupNorm( num_groups=1, num_channels=num_hidden, eps=1e-5, affine=True)
            self.norm3 = nn.GroupNorm( num_groups=1, num_channels=num_hidden, eps=1e-5, affine=True)

        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.dropout3 = nn.Dropout(p=dropout_p)
        self.projection = Linear(num_hidden, num_output)

    def forward(self, input_):
        input_ = input_.transpose(1, 2) 
        input_ = self.dropout1(t.relu(self.norm1(self.conv1(input_)))) 
        input_ = self.dropout2(t.relu(self.norm2(self.conv2(input_)))) 
        input_ = self.dropout3(t.relu(self.norm3(self.conv3(input_)))) 
        input_ = input_.transpose(1, 2) 
        input_ = self.projection(input_) 

        return input_


class Encoder(nn.Module):
    """
    Encoder Network
    """
    def __init__(self, num_hidden=128, num_layers=3, num_head=4, pre_layers=True, pre_norm='batch_norm'):
        """
        :param num_hidden: dimension of hidden
        """
        super(Encoder, self).__init__()
        if pre_layers:
            self.pre_layers = EncoderPrenet( num_hidden, num_hidden*2, num_hidden, dropout_p=0.2, norm_func=pre_norm)
        else:
            self.pre_layers = None
        self.alpha = nn.Parameter(t.ones(1))
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(1024, num_hidden, padding_idx=0),
                                                    freeze=True)
        self.pos_dropout = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm(num_hidden)
        self.layers = clones(Attention(num_hidden, h=num_head, concat_after=False), num_layers)
        self.ffns = clones(FFN(num_hidden), num_layers)

    def forward(self, x, pos):

        # Encoder pre-network
        if self.pre_layers is not None:
            x = self.pre_layers(x)

        # Get character mask
        if self.training:
            c_mask = pos.ne(0).type(t.float)
            mask = pos.eq(0).unsqueeze(1).repeat(1, x.size(1), 1)
        else:
            c_mask, mask = None, None                

        # Get positional embedding, apply alpha and add
        pos = self.pos_emb(pos)
        x = pos * self.alpha + x

        # Positional dropout
        x = self.norm(self.pos_dropout(x))

        # Attention encoder-encoder
        attns = list()
        for layer, ffn in zip(self.layers, self.ffns):
            x, attn = layer(x, x, mask=mask, query_mask=c_mask)
            x = ffn(x)
            attns.append(attn)

        return x, c_mask


class DecoderPostNet(nn.Module):
    """
    Post Convolutional Network (mel --> mel) for decoder
    """
    def __init__(self, num_mels, num_hidden, outputs_per_step=1, use_weight_norm=True, norm_func='batch_norm'):
        """
        
        :param num_hidden: dimension of hidden 
        """
        super(DecoderPostNet, self).__init__()
        self.conv1 = Conv(in_channels=num_mels * outputs_per_step,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=4,
                          w_init='tanh')
        self.conv_list = clones(Conv(in_channels=num_hidden,
                                     out_channels=num_hidden,
                                     kernel_size=5,
                                     padding=4,
                                     w_init='tanh'), 3)
        self.conv2 = Conv(in_channels=num_hidden,
                          out_channels=num_mels * outputs_per_step,
                          kernel_size=5,
                          padding=4)

        if norm_func == 'batch_norm':
            self.norm_list = clones(nn.BatchNorm1d(num_hidden), 3)
            self.pre_norm = nn.BatchNorm1d(num_hidden)
        else:
            self.norm_list = clones( nn.GroupNorm( num_groups=1, num_channels=num_hidden, eps=1e-5, affine=True), 3)
            self.pre_norm = nn.GroupNorm( num_groups=1, num_channels=num_hidden, eps=1e-5, affine=True)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout_list = nn.ModuleList([nn.Dropout(p=0.1) for _ in range(3)])

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, input_, mask=None):
        # Causal Convolution (for auto-regressive)
        input_ = self.dropout1(t.tanh(self.pre_norm(self.conv1(input_)[:, :, :-4])))
        for norm_layer, conv, dropout in zip(self.norm_list, self.conv_list, self.dropout_list):
            input_ = dropout(t.tanh(norm_layer(conv(input_)[:, :, :-4])))
        input_ = self.conv2(input_)[:, :, :-4]
        return input_

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return
        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.utils.weight_norm(m)
        self.apply(_apply_weight_norm)


class Decoder(nn.Module):
    """
    Decoder Network
    """
    def __init__(self,  num_mels=80, 
                        num_hidden=128, 
                        num_condition=128, 
                        num_layers=3, 
                        outputs_per_step=1, 
                        num_head=4,
                        pre_layer=True,
                        post_norm='batch_norm'):
        """
        :param num_hidden: dimension of hidden
        """
        super(Decoder, self).__init__()
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(1024, num_hidden, padding_idx=0),
                                                    freeze=True)
        self.pos_dropout = nn.Dropout(p=0.1)
        self.alpha = nn.Parameter(t.ones(1))
        
        if pre_layer:
            self.pre_layer = DecoderPrenet( num_mels, num_hidden * 2, num_hidden, p=0.2)
        else:
            self.pre_layer = Linear( num_mels, num_hidden)

        self.norm = nn.LayerNorm(num_hidden)

        self.selfattn_layers = clones(Attention(num_hidden, h=num_head, concat_after=False), num_layers)
        self.dotattn_layers = clones(Attention(num_hidden, h=num_head, num_condition=num_condition, concat_after=False), num_layers)
        self.ffns = clones(FFN(num_hidden), num_layers)
        
        self.mel_linear = Linear(num_hidden, num_mels * outputs_per_step)
        self.stop_linear = Linear(num_hidden, 1, w_init='sigmoid')

        self.postconvnet = DecoderPostNet( num_mels, num_hidden, norm_func=post_norm)


    def forward(self, memory, decoder_input, c_mask, pos):
        batch_size = memory.size(0)
        decoder_len = decoder_input.size(1)

        # get decoder mask with triangular matrix
        if self.training:
            m_mask = pos.ne(0).type(t.float)
            mask = m_mask.eq(0).unsqueeze(1).repeat(1, decoder_len, 1)
            if next(self.parameters()).is_cuda:
                mask = mask + t.triu(t.ones(decoder_len, decoder_len).cuda(), diagonal=1).repeat(batch_size, 1, 1).byte()
            else:
                mask = mask + t.triu(t.ones(decoder_len, decoder_len), diagonal=1).repeat(batch_size, 1, 1).byte()
            mask = mask.gt(0)
            zero_mask = c_mask.eq(0).unsqueeze(-1).repeat(1, 1, decoder_len)
            zero_mask = zero_mask.transpose(1, 2)
        else:
            if next(self.parameters()).is_cuda:
                mask = t.triu(t.ones(decoder_len, decoder_len).cuda(), diagonal=1).repeat(batch_size, 1, 1).byte()
            else:
                mask = t.triu(t.ones(decoder_len, decoder_len), diagonal=1).repeat(batch_size, 1, 1).byte()
            mask = mask.gt(0)
            m_mask, zero_mask = None, None

        # Centered position
        if self.pre_layer is not None:
            decoder_input = self.pre_layer(decoder_input)

        # Get positional embedding, apply alpha and add
        pos = self.pos_emb(pos)
        decoder_input = pos * self.alpha + decoder_input

        # Positional dropout
        decoder_input = self.norm(self.pos_dropout(decoder_input))

        # Attention decoder-decoder, encoder-decoder
        attn_dot_list = list()
        attn_dec_list = list()

        for selfattn, dotattn, ffn in zip(self.selfattn_layers, self.dotattn_layers, self.ffns):
            decoder_input, attn_dec = selfattn(decoder_input, decoder_input, mask=mask, query_mask=m_mask)
            decoder_input, attn_dot = dotattn(memory, decoder_input, mask=zero_mask, query_mask=m_mask)
            decoder_input = ffn(decoder_input)
            attn_dot_list.append(attn_dot)
            attn_dec_list.append(attn_dec)

        # Mel linear projection
        mel_out = self.mel_linear(decoder_input)

        # Post Mel Network
        postnet_input = mel_out.transpose(1, 2)
        out = self.postconvnet(postnet_input)
        out = postnet_input + out
        out = out.transpose(1, 2)

        # Stop tokens
        stop_tokens = t.sigmoid(self.stop_linear(decoder_input))

        return mel_out, out, stop_tokens


def collate(data):
    wid, spk_id, pos = zip(*data)
    pos = t.cat([p.unsqueeze(0) for p in pos],dim=0)
    spk_id = t.cat([s.unsqueeze(0) for s in spk_id],dim=0)
    max_length = pos.ne(0).sum(dim=-1).max()
    pos = pos[:,:max_length]
    wid = t.cat([w[:max_length].unsqueeze(0) for w in wid],dim=0)
    return wid, spk_id, pos
