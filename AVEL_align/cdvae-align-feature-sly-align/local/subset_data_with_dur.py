import os
import numpy as np
from scipy.io import wavfile
from pathlib import Path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str,
					help='directory to input list')	
parser.add_argument('out_dir', type=str,
					help='directory to output list')	
parser.add_argument('max_duration', type=float,
					help='Maximum duration of each utterence')
parser.add_argument('-m','--min_duration', type=float, default=0.0,
					help='Minimum duration of each utterence')
args = parser.parse_args()

data_dir = Path(args.data_dir)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

max_duration = args.max_duration
min_duration = args.min_duration

with open(data_dir / 'wav.scp','r') as rf:
	utt2path = [line.rstrip().split() for line in rf.readlines()]
with open(data_dir / 'utt2spk','r') as rf:
	utt2spk = dict([line.rstrip().split() for line in rf.readlines()])
with open(data_dir / 'utt2spk_id','r') as rf:
	utt2spk_id = dict([line.rstrip().split() for line in rf.readlines()])

utt2path_new = list()
utt2spk_new = list()
utt2spk_id_new = list()
utt2dur_new = list()

for utt,path in utt2path:
	fs, x = wavfile.read(path)
	dur = round(len(x)/fs,3)

	if min_duration <= dur and dur <= max_duration:
		utt2path_new.append([utt,path])
		utt2spk_new.append([utt,utt2spk[utt]])
		utt2spk_id_new.append([utt,utt2spk_id[utt]])
		utt2dur_new.append([utt,dur])


with open(out_dir / 'wav.scp','w') as wf:
	for utt,path in utt2path_new:
		wf.write('{} {}\n'.format(utt,path))

with open(out_dir / 'utt2spk','w') as wf:
	for utt,spk in utt2spk_new:
		wf.write('{} {}\n'.format(utt,spk))

with open(out_dir / 'utt2spk_id','w') as wf:
	for utt,spk_id in utt2spk_id_new:
		wf.write('{} {}\n'.format(utt,spk_id))

with open(out_dir / 'utt2dur','w') as wf:
	for utt,dur in utt2dur_new:
		wf.write('{} {:.3f}\n'.format(utt,dur))		