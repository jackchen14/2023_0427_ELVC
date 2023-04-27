import math
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
EL_parse_dir = Path("EL_parse")
NL_parse_dir = Path("NL_parse")

assert args.feature_kind is not None
feature_kinds = args.feature_kind.split('-')

with open(data_dir / 'feats.scp','r') as rf:
    feat_scp = [line.rstrip().split() for line in rf.readlines()]

# print("test123")
# print("Shape of lista is : "+str(np.shape(feat_scp)))
# for i in range(300):
#     print(i)
#     print(feat_scp[i])


with open(EL_parse_dir / 'EL_parse_timing.scp','r') as rf:
    EL_parse_timing = [line.rstrip().split() for line in rf.readlines()]
    # print("EL_parse_timing")
    # for i in range(len(EL_parse_timing)):
    #     print(i)
    #     EL_parse_timing[i][0] = EL_parse_timing[i][0].split(':')[0]
print(EL_parse_timing[0])

with open(NL_parse_dir / 'NL_parse_timing.scp','r') as rf:
    NL_parse_timing = [line.rstrip().split() for line in rf.readlines()]
    #print("NL_parse_timing")
    #print(len(NL_parse_timing))
    #print(NL_parse_timing)

# print(NL_parse_timing[0])
# print("change")
# print(NL_parse_timing[0][0:10])
    
count = 0
for utt, path in feat_scp:
    feats = torch.load(path)
    key_dict = dict()
    for feat_pair in feature_kinds:
        feat_pair = feat_pair.split(':')
        assert len(feat_pair) == 2

        # import ipdb
        # ipdb.set_trace()
        dtwpath = np.array([]).reshape(2,0)
        feat1 = feats[feat_pair[0]].t().numpy().copy(order='C')
        ############# feat_pair[0] == nl_mcc
        #print("nlmcc frames num = ")
        #print(len(feat1))
        feat2 = feats[feat_pair[1]].t().numpy().copy(order='C')
        ############# feat_pair[1] == el_mcc
        for i in range(len(NL_parse_timing)):
            if (NL_parse_timing[i][0] == utt):
                temp_idx = i
                break
        print("current file utt = " , utt )
        print("current file nl lenth = " , len(feat1) )
        print("current file el lenth = " , len(feat2) )
        print("current file parse_timing = " , NL_parse_timing[temp_idx][0])
        for i in range(int((len(NL_parse_timing[temp_idx])-1)/3)):
            if (NL_parse_timing[temp_idx][3*i+1] == EL_parse_timing[temp_idx][3*i+1]):
                #print("good QQ")
                print("nl idx")
                temp1_start = float(NL_parse_timing[temp_idx][3*i+2])
                print("temp1_start = " ,temp1_start)
                temp1_start_frame = ((temp1_start * 16000) - 1024)/256
                temp1_stop = float(NL_parse_timing[temp_idx][3*i+3])
                print("temp1_stop = " ,temp1_stop)
                temp1_stop_frame = ((temp1_stop * 16000) - 1024)/256
                if (temp1_start_frame <= 0):
                    NL_start_time_idx = 0
                    print(NL_start_time_idx)
                else:
                    NL_start_time_idx = math.ceil(temp1_start_frame)
                    print(NL_start_time_idx)
                if (temp1_stop_frame <= 0):
                    NL_stop_time_idx = 0
                    print(NL_stop_time_idx)
                else: 
                    NL_stop_time_idx = math.ceil(temp1_stop_frame)
                    print(NL_stop_time_idx)
                print("el idx")
                temp1_start = float(EL_parse_timing[temp_idx][3*i+2])
                temp1_start_frame = ((temp1_start * 16000) - 1024)/256
                temp1_stop = float(EL_parse_timing[temp_idx][3*i+3])
                temp1_stop_frame = ((temp1_stop * 16000) - 1024)/256
                if (temp1_start_frame <= 0):
                    EL_start_time_idx = 0
                    print(EL_start_time_idx)
                else:
                    EL_start_time_idx = math.ceil(temp1_start_frame)
                    print(EL_start_time_idx)
                if (temp1_stop_frame <= 0):
                    EL_stop_time_idx = 0
                    print(EL_stop_time_idx)
                else: 
                    EL_stop_time_idx = math.ceil(temp1_stop_frame)
                    print(EL_stop_time_idx)
            else :
                print("bad")
            seg_nl = feat1[NL_start_time_idx:(NL_stop_time_idx)]
            seg_el = feat2[EL_start_time_idx:(EL_stop_time_idx)]
            temp_dtwpath = estimate_twf( seg_nl, seg_el, distance='euclidean', unique=int(args.unique_side))
            # print(temp_dtwpath.shape)
            # print(dtwpath.shape)
            temp_dtwpath[0] = temp_dtwpath[0] + NL_start_time_idx ## shift
            temp_dtwpath[1] = temp_dtwpath[1] + EL_start_time_idx ## shift

            dtwpath = np.append(dtwpath , temp_dtwpath ,axis = 1)
            dtwpath = dtwpath.astype(int)
            
        
        # print("count = ")
        # print(count)
        print("dtwpath")
        print(type(dtwpath))
        print(dtwpath)
        print("dtwpath[0][0]")
        print(dtwpath[0][0])
        print(type(dtwpath[0][0]))
        # print(dtwpath.shape)
        
        feats['dtw_' + feat_pair[0]] = torch.from_numpy(dtwpath[0]).long()
        feats['dtw_' + feat_pair[1]] = torch.from_numpy(dtwpath[1]).long()
        
        print(feats['dtw_' + feat_pair[0]])
        # np.save("./sly_aligned_NL_array/" + NL_parse_timing[temp_idx][0] + ".npy", feats['nl_mcc'])
        # np.save("./sly_aligned_EL_array/" + NL_parse_timing[temp_idx][0] + ".npy", feats['el_mcc'])
        if (NL_parse_timing[temp_idx][0] == "NL01_001"):
            temp_test_dtw_nl_mcc = feats['dtw_nl_mcc']
            temp_test_dtw_el_mcc = feats['dtw_el_mcc']
        np.save("./sly_aligned_NL_array/" + NL_parse_timing[temp_idx][0] + ".npy", feats['dtw_' + feat_pair[0]])
        np.save("./sly_aligned_EL_array/" + NL_parse_timing[temp_idx][0] + ".npy", feats['dtw_' + feat_pair[1]])
        print(feats)
        count += 1

    torch.save(feats,path)
print("dtw_nl_mcc")
print(temp_test_dtw_nl_mcc.shape)
print(temp_test_dtw_nl_mcc)
print("dtw_el_mcc")
print(temp_test_dtw_el_mcc.shape)
print(temp_test_dtw_el_mcc)






