import os
import numpy as np
from pathlib import Path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str,
					help='directory to input list')	
parser.add_argument('out_dir', type=str,
					help='directory to output list')	
parser.add_argument('num_utt_per_spk', type=int,
					help='Number of utterences per speaker')
args = parser.parse_args()

data_dir = Path(args.data_dir)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

num_utt_per_spk = args.num_utt_per_spk

with open(data_dir / 'wav.scp','r') as rf:
	utt2path = dict([line.rstrip().split() for line in rf.readlines()])
with open(data_dir / 'utt2spk','r') as rf:
	utt2spk = dict([line.rstrip().split() for line in rf.readlines()])
with open(data_dir / 'utt2spk_id','r') as rf:
	utt2spk_id = dict([line.rstrip().split() for line in rf.readlines()])
with open(data_dir / 'spk2utt','r') as rf:
	spk2utt = [line.rstrip().split() for line in rf.readlines()]

utt2path_new = list()
utt2spk_new = list()
utt2spk_id_new = list()

for line in spk2utt:
	spk = line[0]
	for num_utt, utt in enumerate(line[1:]):
		if num_utt >= num_utt_per_spk:
			break

		utt2path_new.append([utt,utt2path[utt]])
		utt2spk_new.append([utt,spk])
		utt2spk_id_new.append([utt,utt2spk_id[utt]])


with open(out_dir / 'wav.scp','w') as wf:
	for utt,path in utt2path_new:
		wf.write('{} {}\n'.format(utt,path))

with open(out_dir / 'utt2spk','w') as wf:
	for utt,spk in utt2spk_new:
		wf.write('{} {}\n'.format(utt,spk))

with open(out_dir / 'utt2spk_id','w') as wf:
	for utt,spk_id in utt2spk_id_new:
		wf.write('{} {}\n'.format(utt,spk_id))