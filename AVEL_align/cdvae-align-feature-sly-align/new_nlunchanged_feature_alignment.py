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
    for i in feats :
        print(i)
    # ipdb.set_trace()
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
        print("current file el lenth = " , len(feat1) )
        print("current file nl lenth = " , len(feat2) )
        print("current file parse_timing = " , NL_parse_timing[temp_idx][0])

        last_EL_stop_time_idx = 0
        last_array = []
        # ipdb.set_trace()
        WavLM_frame_size = 400
        WavLM_hop_size = 320
        for i in range(int((len(NL_parse_timing[temp_idx])-1)/3)):
            if (NL_parse_timing[temp_idx][3*i+1] == EL_parse_timing[temp_idx][3*i+1]):
                #print("good QQ")
                print("nl idx")
                temp1_start = float(NL_parse_timing[temp_idx][3*i+2])
                print("temp1_start = " ,temp1_start)
                temp1_start_frame = ((temp1_start * 16000) - WavLM_frame_size)/WavLM_hop_size
                temp1_stop = float(NL_parse_timing[temp_idx][3*i+3])
                print("temp1_stop = " ,temp1_stop)
                temp1_stop_frame = ((temp1_stop * 16000) - WavLM_frame_size)/WavLM_hop_size
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
                temp1_start_frame = ((temp1_start * 16000) - WavLM_frame_size)/WavLM_hop_size
                print("temp1_start = " ,temp1_start)
                print("check length")
                print(len(EL_parse_timing[temp_idx]))
                if 3*i+5 < len(EL_parse_timing[temp_idx]) :
                    temp1_stop = float(EL_parse_timing[temp_idx][3*i+5])
                    print("in range")
                else :
                    temp1_stop = float(EL_parse_timing[temp_idx][3*i+3])  
                    print("out of range")
                print("temp1_start = " ,temp1_stop)          
                
                temp1_stop_frame = ((temp1_stop * 16000) - WavLM_frame_size)/WavLM_hop_size
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
            print(feat1.shape[0])
            feat1_len = feat1.shape[0]
            feat2_len = feat2.shape[0]

            if (NL_stop_time_idx >= feat2_len) :
                NL_stop_time_idx = feat2_len-1  
            if (EL_stop_time_idx >= feat1_len) :
                EL_stop_time_idx = feat1_len-1     

            seg_el = feat1[EL_start_time_idx:(EL_stop_time_idx)]
            seg_nl = feat2[NL_start_time_idx:(NL_stop_time_idx)]
            temp_dtwpath = estimate_twf( seg_el, seg_nl, distance='euclidean', unique=int(args.unique_side))
            # print(temp_dtwpath.shape)
            # print(dtwpath.shape)
            temp_dtwpath[0] = temp_dtwpath[0] + EL_start_time_idx ## shift
            temp_dtwpath[1] = temp_dtwpath[1] + NL_start_time_idx ## shift
            dtwpath = np.append(dtwpath , temp_dtwpath ,axis = 1)
            dtwpath = dtwpath.astype(int)

            #### after this for loop is over
            #### you will obtain el index of a complete sentence (utterance)
            #### you will obtain nl index of a complete sentence (utterance)

            
        
 
        # ipdb.set_trace()   
        
        # print(dtwpath.shape)
        
        ### feat1 = feats[feat_pair[0]].t().numpy().copy(order='C')
        # feats['dtw_' + feat_pair[0]] = torch.from_numpy(dtwpath[0]).long()
        # feats['dtw_' + feat_pair[1]] = torch.from_numpy(dtwpath[1]).long()
        print(type(feats))
        print("--------orginal feats----------")
        for i in feats :
            print(i , feats[i].shape)

        ##  transfer wavlm index to wld index
        wld_feature_dtwpath = copy.deepcopy(dtwpath)

        temp_nl_count = 0
        for i in dtwpath[0]:
            a = ((400+(i-1)*320)-1024)/256
            if (a <= 0):
                wld_idx = 0
                wld_feature_dtwpath[0][temp_nl_count] = wld_idx
            else:
                wld_idx = math.ceil(a)
                wld_feature_dtwpath[0][temp_nl_count] = wld_idx
            temp_nl_count += 1
        
        temp_el_count = 0
        for i in dtwpath[1]:
            a = ((400+(i-1)*320)-1024)/256
            if (a <= 0):
                wld_idx = 0
                wld_feature_dtwpath[1][temp_el_count] = wld_idx
            else:
                wld_idx = math.ceil(a)
                wld_feature_dtwpath[1][temp_el_count] = wld_idx
            temp_el_count += 1

        print("-------- wavlm index --------")
        print("nl_idx")
        print(dtwpath[1])
        print("el_idx")
        print(dtwpath[0])
        # ##error testing
        #### complete_el_array = wld_feature_dtwpath[1]
        #### complete_nl_array = wld_feature_dtwpath[0]
        # ##error testing    
        print("-------- direct transfer wavlm index to wld index --------")
        print("nl_idx")
        print(wld_feature_dtwpath[1])
        print("el_idx")
        print(wld_feature_dtwpath[0])

        ##### remove duplicate index 0 in the begin of wld_feature_dtwpath #####
        remove_idx = []
        for i in range(len(wld_feature_dtwpath[1])):   ## here
            if (i!=0 and wld_feature_dtwpath[1][i] == 0): ## here
                remove_idx.append(i)

        wld_feature_dtwpath = np.delete(wld_feature_dtwpath ,remove_idx ,axis=1)

        print("-------- delete duplicated zero at begining --------")
        print("nl_idx(no zero)")
        print(wld_feature_dtwpath[1])
        print("el_idx(no zero)")
        print(wld_feature_dtwpath[0])

        # ipdb.set_trace()
        ##### make el wld_feature_dtwpath complete #####
        start = wld_feature_dtwpath[1][0]
        stop = wld_feature_dtwpath[1][-1]
        length = stop - start + 1
        complete_nl_array = np.empty([length])
        for i in range(length):
             complete_nl_array[i] = start + i

        ##### make nl wld_feature_dtwpath complete according to past idx (Ney Xa Fa)#####
        count_a = 0
        count_b = 0
        complete_el_array = np.empty([length])
        for i in range(length):
            if (wld_feature_dtwpath[1][count_a] == i + start) :
                complete_el_array[i] = wld_feature_dtwpath[0][count_b]
                count_a += 1
                count_b += 1
            else :
                complete_el_array[i] = math.floor((wld_feature_dtwpath[0][count_b-1] + wld_feature_dtwpath[0][count_b])/2)

        complete_el_array = complete_el_array.astype('int')
        complete_nl_array = complete_nl_array.astype('int')
        print("-------- After Ney Xa Fa --------")      
        print("new nl array")
        print(complete_nl_array.shape)
        print(complete_nl_array)
        print("new el array")
        print(complete_el_array.shape)
        print(complete_el_array)

        # ipdb.set_trace()
        ## create temp dictionary ##
        temp_feats = {}
        for i in feats :
            if i.split('_')[1]== 'WavLM':
                if i.split('_')[0] == 'nl':
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
                elif i.split('_')[0] == 'el':
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
            else :
                if i.split('_')[0] == 'nl':
                    temp = np.empty([len(feats[i][:,0]),len(complete_nl_array)])
                    count = 0
                    for j in complete_nl_array :
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
                    temp = np.empty([len(feats[i][:,0]),len(complete_el_array)])
                    count = 0
                    for j in complete_el_array :
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
        if (NL_parse_timing[temp_idx][0] == "NL01_001"):
            temp_test_dtw_nl_mcc = feats['dtw_nl_mcc']
            temp_test_dtw_el_mcc = feats['dtw_el_mcc']
        
        ## save dtw data as np
        # np.save("./sly_aligned_NL_array/" + NL_parse_timing[temp_idx][0] + ".npy", feats['dtw_' + feat_pair[0]])
        # np.save("./sly_aligned_EL_array/" + NL_parse_timing[temp_idx][0] + ".npy", feats['dtw_' + feat_pair[1]])

        ## save dtw index as np
        np.save("./sly_aligned_NL_array/" + NL_parse_timing[temp_idx][0] + ".npy", complete_nl_array)
        np.save("./sly_aligned_EL_array/" + NL_parse_timing[temp_idx][0] + ".npy", complete_el_array)

    torch.save(feats,path)

# print("avg_distance score")
# avg_distance_score = distance_score / distance_count
# print(avg_distance_score)


print("dtw_nl_mcc")
print(temp_test_dtw_nl_mcc.shape)
print(temp_test_dtw_nl_mcc)
print("dtw_el_mcc")
print(temp_test_dtw_el_mcc.shape)
print(temp_test_dtw_el_mcc)







