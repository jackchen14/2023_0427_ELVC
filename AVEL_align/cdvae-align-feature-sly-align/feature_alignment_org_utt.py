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
    print("feat_scp")
    print(feat_scp)

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
        # print("count = ")
        # print(count)
        
        ## create temp dictionary ##
        temp_feats = {}
        ## insert index
        for i in feats :
            if i.split('_')[0] == 'nl':
                temp = np.empty([len(feats[i][:,0]),len(dtwpath[0])])
                print(temp.shape)
                print("------------")
                print(feats[i].shape)
                count = 0
                for j in dtwpath[0] :
                    temp[:,count] = feats[i][:,j]
                    count += 1
                temp_feats['dtw_' + i] = torch.from_numpy(temp).float()
            elif i.split('_')[0] == 'el':
                temp = np.empty([len(feats[i][:,0]),len(dtwpath[1])])
                count = 0
                for j in dtwpath[1] :
                    temp[:,count] = feats[i][:,j]
                    count += 1
                temp_feats['dtw_' + i] = torch.from_numpy(temp).float()

        for i in temp_feats :
            feats[i] = temp_feats[i]
        print("--------new feats----------")
        for i in feats :
            print(i , feats[i].shape)


        # print(feats['dtw_' + feat_pair[0]])
        # np.save("./sly_aligned_NL_array/" + NL_parse_timing[temp_idx][0] + ".npy", dtwpath[0])
        # np.save("./sly_aligned_EL_array/" + NL_parse_timing[temp_idx][0] + ".npy", dtwpath[1])
        # count += 1

    torch.save(feats,path)




