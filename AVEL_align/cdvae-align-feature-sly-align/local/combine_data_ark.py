from pathlib import Path
import torch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('out_scp_dir', type=str,
                    help='directory to output list')
parser.add_argument('out_ark_dir', type=str,
                    help='directory to output ark.')
parser.add_argument('data_dir', type=str,
                    help='directory to input list')
parser.add_argument('-K', '--feature_kind', type=str, default=None,
                    help='Feature kind')
args = parser.parse_args()

out_scp_dir = Path(args.out_scp_dir)
out_scp_dir.mkdir(parents=True, exist_ok=True)
out_ark_dir = Path(args.out_ark_dir)
out_ark_dir.mkdir(parents=True, exist_ok=True)
data_dirs = [d.split(':') for d in args.data_dir.split(',')]

feature_kinds = None if args.feature_kind is None else args.feature_kind.split('-')

feats_dict = list()
feats_list = list()

for data_name, data_dir in data_dirs:
    data_dir = Path(data_dir)
    with open(data_dir / 'feats.scp','r') as rf:
        feats_dict.append([
                data_name,
                dict([line.rstrip().split() for line in rf.readlines()])
            ])

    for utt in feats_dict[-1][1].keys():
        if utt not in feats_list:
            feats_list.append(utt)

feats_list = sorted(feats_list)


list_idxs=[int(idx.split('_')[1]) for idx in feats_list]
list_idxs=list(set(list_idxs))


def pair_feats(out_feats, in_feats, cur_id, cur_utt):


    for key, val in in_feats.items():
        if feature_kinds is not None and key not in feature_kinds:
            continue
        key_new = cur_id + '_' + key
        print(f'{cur_utt}, new key:{key_new}, data_name: {data_name}')
        out_feats[key_new] = val

    return out_feats

el_id, el_dict = feats_dict[0]
nl_id, nl_dict = feats_dict[1]

with open(out_scp_dir / 'feats.scp', 'w') as wf:

    for cur_utt_idx in list_idxs:

        cur_feats=dict()
        cur_utt=f'NL01_{cur_utt_idx:03d}'
        file_feats=Path(out_ark_dir, f'{cur_utt}.pt')
        
        el_feats = torch.load(el_dict[f'EL01_{cur_utt_idx:03d}'], map_location='cpu')    
        nl_feats = torch.load(nl_dict[f'NL01_{cur_utt_idx:03d}'], map_location='cpu')

        cur_feats = pair_feats(cur_feats, el_feats, el_id, cur_utt)
        cur_feats = pair_feats(cur_feats, nl_feats, nl_id, cur_utt)
        for dictionary_name in cur_feats :
            print(dictionary_name)
            print(cur_feats[dictionary_name].shape)

        torch.save(cur_feats, file_feats)
        wf.write(f'{cur_utt} {file_feats}\n')
