import math
import os
import numpy as np
import torch
from util.distance import estimate_twf
from pathlib import Path
import ipdb
import copy


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
    
distance_score = 0
distance_count = 0
for utt, path in feat_scp:
    feats = torch.load(path)
    # print("--------orginal feats----------")
    # for i in feats :
    #     print(i , feats[i].shape)
    # ipdb.set_trace()
    key_dict = dict()
    for feat_pair in feature_kinds:
        feat_pair = feat_pair.split(':')
        assert len(feat_pair) == 2
        # print("--------orginal feats----------")
        # # for i in feats :
        # #     if(i.split("_")[-1] == "WavLM") :
        # #         # print(feats[i][:,0])
        # #         start = copy.deepcopy(feats[i][:,0])
        # #         start = torch.reshape(start,(768,1))
        # #         stop = copy.deepcopy(feats[i][:,-1])
        # #         stop = torch.reshape(stop,(768,1))
        # #         # print(start)
        # #         feats[i] = torch.cat((start,feats[i]),1)
        # #         feats[i] = torch.cat((feats[i],stop),1)
        # for i in feats :
        #     print(i , feats[i].shape)
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
        print("-----"+utt+"------")
        print("current file utt = " , utt )
        print("current file nl lenth = " , len(feat1) )
        print("current file el lenth = " , len(feat2) )
        print("current file parse_timing = " , NL_parse_timing[temp_idx][0])

        # ipdb.set_trace()
        WavLM_frame_size = 400
        WavLM_hop_size = 320
        for i in range(int((len(NL_parse_timing[temp_idx])-1)/3)):
            if (NL_parse_timing[temp_idx][3*i+1] == EL_parse_timing[temp_idx][3*i+1]):
                print("nl idx")
                temp1_start = float(NL_parse_timing[temp_idx][3*i+2])
                print("temp1_start = " ,temp1_start)
                temp1_start_frame = ((temp1_start * 16000) - WavLM_frame_size)/WavLM_hop_size
                if 3*i+5 < len(NL_parse_timing[temp_idx]) :
                    temp1_stop = float(NL_parse_timing[temp_idx][3*i+5])
                else :
                    temp1_stop = float(NL_parse_timing[temp_idx][3*i+3])

                print("temp1_stop = " ,temp1_stop)
                temp1_stop_frame = ((temp1_stop * 16000) - WavLM_frame_size)/WavLM_hop_size
                if (temp1_start_frame <= 0):
                    NL_start_time_idx = 0
                else:
                    NL_start_time_idx = math.ceil(temp1_start_frame)
                
                if (temp1_stop_frame <= 0):
                    NL_stop_time_idx = 0
                else: 
                    NL_stop_time_idx = math.ceil(temp1_stop_frame)


                print("el idx")
                temp1_start = float(EL_parse_timing[temp_idx][3*i+2])
                temp1_start_frame = ((temp1_start * 16000) - WavLM_frame_size)/WavLM_hop_size
                print("el_temp1_start = " ,temp1_start)
                print(len(EL_parse_timing[temp_idx]))
                if 3*i+5 < len(EL_parse_timing[temp_idx]) :
                    temp1_stop = float(EL_parse_timing[temp_idx][3*i+5])
                else :
                    temp1_stop = float(EL_parse_timing[temp_idx][3*i+3])  
                print("el_temp1_stop = " ,temp1_stop)          
                
                temp1_stop_frame = ((temp1_stop * 16000) - WavLM_frame_size)/WavLM_hop_size
                if (temp1_start_frame <= 0):
                    EL_start_time_idx = 0
                else:
                    EL_start_time_idx = math.ceil(temp1_start_frame)
                if (temp1_stop_frame <= 0):
                    EL_stop_time_idx = 0
                else: 
                    EL_stop_time_idx = math.ceil(temp1_stop_frame)
            else :
                print("bad")
            
            feat1_len = len(feat1)
            feat2_len = len(feat2)

            if (NL_stop_time_idx >= feat1_len) :
                NL_stop_time_idx = feat1_len-1  
            if (EL_stop_time_idx >= feat2_len) :
                EL_stop_time_idx = feat2_len-1     

            seg_nl = feat1[NL_start_time_idx:(NL_stop_time_idx)]
            seg_el = feat2[EL_start_time_idx:(EL_stop_time_idx)]
            temp_dtwpath = estimate_twf( seg_nl, seg_el, distance='euclidean', unique=int(args.unique_side))
            # print(temp_dtwpath.shape)
            # print(dtwpath.shape)
            temp_dtwpath[0] = temp_dtwpath[0] + NL_start_time_idx ## shift
            temp_dtwpath[1] = temp_dtwpath[1] + EL_start_time_idx ## shift
            dtwpath = np.append(dtwpath , temp_dtwpath ,axis = 1)
            dtwpath = dtwpath.astype(int)

        
        # ipdb.set_trace()   
        print("dtwpath")
        print(type(dtwpath))
        print(dtwpath[1])
        
        # print(dtwpath.shape)
        
        ### feat1 = feats[feat_pair[0]].t().numpy().copy(order='C')
        # feats['dtw_' + feat_pair[0]] = torch.from_numpy(dtwpath[0]).long()
        # feats['dtw_' + feat_pair[1]] = torch.from_numpy(dtwpath[1]).long()
        print(type(feats))
        print("--------orginal feats----------")
        for i in feats :
            print(i , feats[i].shape)
        
        ## create temp dictionary ##
        temp_feats = {}
        for i in feats :
            if i.split('_')[0] == 'nl':
                temp = np.empty([len(feats[i][:,0]),len(dtwpath[0])])
                count = 0
                for j in dtwpath[0] :
                    temp[:,count] = feats[i][:,j]
                    count += 1
                temp_feats['dtw_' + i] = torch.from_numpy(temp).float()
                # print("123")
                # print(len(dtwpath[0]))
                # print(i)
                # print(temp_feats['dtw_' + i].shape)
                # print(feats[i].shape)
                # ipdb.set_trace()
            elif i.split('_')[0] == 'el':
                temp = np.empty([len(feats[i][:,0]),len(dtwpath[1])])
                count = 0
                for j in dtwpath[1] :
                    temp[:,count] = feats[i][:,j]
                    count += 1
                temp_feats['dtw_' + i] = torch.from_numpy(temp).float()
                # print("123")
                # print(len(dtwpath[0]))
                # print(i)
                # print(temp_feats['dtw_' + i].shape)
                # print(feats[i].shape)
                # ipdb.set_trace()      

        for i in temp_feats :
            feats[i] = temp_feats[i]
        print("--------new feats----------")
        for i in feats :
            print(i , feats[i].shape)

        # ipdb.set_trace()
        ####### dtw score #######
        # temp_score = estimate_twf.distance_func(feats['dtw_el_mcc'], feats['dtw_nl_mcc'])
        # feats[feat_pair[0]].t().numpy().copy(order='C')
        # temp_score = np.mean((feats['dtw_el_mcc'].t().numpy().copy(order='C')-feats['dtw_nl_mcc'].t().numpy().copy(order='C')) ** 2)
        # distance_score = distance_score + temp_score
        # distance_count += 1
        # print(distance_score)
        # print(distance_count)
        ####### dtw score #######

        # np.save("./sly_aligned_NL_array/" + NL_parse_timing[temp_idx][0] + ".npy", feats['nl_mcc'])
        # np.save("./sly_aligned_EL_array/" + NL_parse_timing[temp_idx][0] + ".npy", feats['el_mcc'])
        # if (NL_parse_timing[temp_idx][0] == "NL01_001"):
        #     temp_test_dtw_nl_mcc = feats['dtw_nl_mcc']
        #     temp_test_dtw_el_mcc = feats['dtw_el_mcc']
        
        ## save dtw data as np
        # np.save("./sly_aligned_NL_array/" + NL_parse_timing[temp_idx][0] + ".npy", feats['dtw_' + feat_pair[0]])
        # np.save("./sly_aligned_EL_array/" + NL_parse_timing[temp_idx][0] + ".npy", feats['dtw_' + feat_pair[1]])

        ## save dtw index as np
        np.save("./sly_aligned_NL_array/" + NL_parse_timing[temp_idx][0] + ".npy", dtwpath[0])
        np.save("./sly_aligned_EL_array/" + NL_parse_timing[temp_idx][0] + ".npy", dtwpath[1])

    torch.save(feats,path)

# print("avg_distance score")
# avg_distance_score = distance_score / distance_count
# print(avg_distance_score)


# print("dtw_nl_mcc")
# print(temp_test_dtw_nl_mcc.shape)
# print(temp_test_dtw_nl_mcc)
# print("dtw_el_mcc")
# print(temp_test_dtw_el_mcc.shape)
# print(temp_test_dtw_el_mcc)







