import os
import numpy as np
from pathlib import Path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str,
					help='directory to the list')
args = parser.parse_args()

data_dir = Path(args.data_dir)

with open(data_dir / 'utt2spk','r') as rf:
	utt2spk = [line.rstrip().split() for line in rf.readlines()]

if not (data_dir / 'spk2spk_id').exists():
	print('Cannot find spk2spk_id. Generate it.')

	spk_list = np.unique([spk for utt,spk in utt2spk])
	spk_list = [[spk,i] for i,spk in enumerate(spk_list)]

	with open(data_dir / 'spk2spk_id','w') as wf:
		for spk,spk_id in spk_list:
			wf.write('{} {}\n'.format(spk,spk_id))

	spk2spk_id = dict(spk_list)
else:
	print('Find spk2spk_id. Use it.')
	with open(data_dir / 'spk2spk_id','r') as rf:
		spk2spk_id = dict([line.rstrip().split() for line in rf.readlines()])


with open(data_dir / 'utt2spk_id','w') as wf:
	for utt,spk in utt2spk:
		wf.write('{} {}\n'.format(utt,spk2spk_id[spk]))




