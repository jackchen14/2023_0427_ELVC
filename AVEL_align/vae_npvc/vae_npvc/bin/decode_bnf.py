#!/usr/bin/env python3

# Copyright 2020 Yu-Huai Peng
# Licensed under the MIT license.
#
# This script masks all the name of the dataset
#

import os
import time
from pathlib import Path
from importlib import import_module

import numpy as np
import math

import torch
import torch.nn.functional as F

from kaldiio import ReadHelper, WriteHelper

MAX_WAV_VALUE = 32768.0


def decode_bnf( args):
    model_path = args.model_path
    bnf_kind = args.bnf_kind
    input_txt = True if args.input_txt.lower() in ['true'] else False

    rspecifier = args.rspecifier
    wspecifier = args.wspecifier

    config = yaml.safe_load(open(args.config))

    model_type = config.get('model_type', 'vae_npvc.model.vqvae')
    module = import_module(model_type, package=None)
    model = getattr(module, 'Model')(config)
    model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
    model.cuda().eval()
    
    if input_txt and bnf_kind in ['id','csid']:
        bnf_reader = open(rspecifier,'r')
    else:
        bnf_reader = ReadHelper(rspecifier)
        input_txt = False

    feat_writer = WriteHelper(wspecifier, compression_method=1)

    if args.utt2spk_id is not None:
        utt2spk_id = dict([l.strip().split(None,1) for l in open(args.utt2spk_id)])
    else:
        utt2spk = [l.strip().split(None,1) for l in open(args.utt2spk)]
        spk2spk_id = dict([l.strip().split(None,1) for l in open(args.spk2spk_id)])
        utt2spk_id = dict([[utt,spk2spk_id[spk]] for utt,spk in utt2spk])

    for line in bnf_reader:
        if input_txt:
            utt, token_id = line.split()
            token_id = [t.split('>')[0] for t in token_id.split('<')]
            token_id = [int(t) for t in token_id if t.isdigit()]
        else:
            utt, token_id = line

        # Load source features
        bnf_in = torch.from_numpy(np.array(token_id))
        bnf_in = bnf_in.long().cuda().unsqueeze(0)
        if bnf_in.ndim >= 3 and bnf_kind != 'token':
            bnf_in = bnf_in.view(1,-1)

        spk_in = torch.tensor(int(utt2spk_id[utt]))
        spk_in = spk_in.long().cuda().view(1,-1)
   
        with torch.no_grad():
            if bnf_kind == 'token':
                bnf_in = model.quantizer.encode(bnf_in)
            feat_out = model.decode((bnf_in,spk_in)).squeeze(0)
            feat_out = feat_out.detach().t().cpu().numpy()

        feat_writer[utt] = feat_out

        print('Decoding BNF {} of {}.'.format( bnf_kind, utt),end=' '*30+'\r')

    print('Finished decoding BNF {}.'.format( bnf_kind),end=' '*len(utt)+'\n')
    bnf_reader.close()
    feat_writer.close()


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/utt2spks.yaml',
                        help='YAML file for configuration')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to checkpoint with model')
    parser.add_argument('--bnf_kind', type=str, default=None,
                        help='Bottleneck feature kinds.')
    parser.add_argument('--input_txt', type=str, default='true')
    parser.add_argument('--utt2spk', type=str, default=None,
                        help='utt2spk file')
    parser.add_argument('--spk2spk_id', type=str, default=None,
                        help='spk2spk_id file')
    parser.add_argument('--utt2spk_id', type=str, default=None,
                        help='utt2spk_id file')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Using gpu #')
    parser.add_argument('rspecifier', type=str,
                        help='Input specifier')
    parser.add_argument('wspecifier', type=str,
                        help='Output specifier')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    decode_bnf(args)



