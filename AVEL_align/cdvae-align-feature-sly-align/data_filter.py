import os
import ipdb
import re
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('el_dir', type=str,
                    help='directory to el data')
parser.add_argument('nl_dir', type=str,
                    help='directory to nl data')
args = parser.parse_args()

# data_dirs = [d.split(':') for d in args.data_dir.split(',')]

el_path = args.el_dir
nl_path = args.nl_dir
dir_list = [el_path,nl_path]
feats_list = list()
for data_dir in dir_list:
    for file in glob.glob(data_dir+'/*'):
        if re.search('_w.wav',file):
            os.remove(file)
        else:
            feats_list.append(file)
feats_list = sorted(feats_list)

list_idxs = [int(idx.split('/')[-1].split('.')[0].split('_')[1]) for idx in feats_list]
# ipdb.set_trace()
list_idxs = list(set([idx for idx in list_idxs if list_idxs.count(idx)==1]))

for idx in list_idxs:
    if os.path.isfile(el_path+f'/EL01_{idx:03d}.wav'):
        os.remove(el_path+f'/EL01_{idx:03d}.wav')
    elif os.path.isfile(nl_path+f'/NL01_{idx:03d}.wav'):
        os.remove(nl_path+f'/NL01_{idx:03d}.wav')
