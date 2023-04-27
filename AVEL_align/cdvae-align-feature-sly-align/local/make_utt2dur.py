import os
import numpy as np
from scipy.io import wavfile
from pathlib import Path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str,
					help='directory to input list')	
args = parser.parse_args()

data_dir = Path(args.data_dir)

with open(data_dir / 'wav.scp','r') as rf:
	utt2path = [line.rstrip().split() for line in rf.readlines()]

utt2dur = list()

for utt,path in utt2path:
	fs, x = wavfile.read(path)
	utt2dur.append([utt,round(len(x)/fs,3),len(x)])

with open(data_dir / 'utt2dur','w') as wf:
	for utt,dur,_ in utt2dur:
		wf.write('{} {:.3f}\n'.format(utt,dur))

import ipdb
ipdb.set_trace()