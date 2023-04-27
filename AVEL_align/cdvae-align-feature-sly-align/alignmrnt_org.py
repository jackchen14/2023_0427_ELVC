import os
import numpy as np
import torch
from util.distance import estimate_twf
from pathlib import Path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str,
                    help='directory to input list')
parser.add_argument('-K', '--feature_kind', type=str, default=None,
                    help='Feature kind')
parser.add_argument('-U', '--unique_side', type=str, default='0',
                    help='Unique side of alignment')
args = parser.parse_args()

data_dir = Path(args.data_dir)

assert args.feature_kind is not None
feature_kinds = args.feature_kind.split('-')

with open(data_dir / 'feats.scp','r') as rf:
    feat_scp = [line.rstrip().split() for line in rf.readlines()]

for utt, path in feat_scp:
    feats = torch.load(path)

    key_dict = dict()
    for feat_pair in feature_kinds:
        feat_pair = feat_pair.split(':')
        assert len(feat_pair) == 2

        # import ipdb
        # ipdb.set_trace()
        feat1 = feats[feat_pair[0]].t().numpy().copy(order='C')
        feat2 = feats[feat_pair[1]].t().numpy().copy(order='C')

        dtwpath = estimate_twf( feat1, feat2, distance='euclidean', unique=int(args.unique_side))

        feats['dtw_' + feat_pair[0]] = torch.from_numpy(dtwpath[0]).long()
        feats['dtw_' + feat_pair[1]] = torch.from_numpy(dtwpath[1]).long()

    torch.save(feats,path)