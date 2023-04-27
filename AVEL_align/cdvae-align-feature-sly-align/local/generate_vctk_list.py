import os
import ipdb
from pathlib import Path
from scipy.io import wavfile


ROOT_PATH = '/mnt/md0/user_roland/VCTK-Corpus/wav24000'
LIST_PATH = 'data'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d','--data_root', type=str, default=ROOT_PATH,
					help='Corpus path')	
parser.add_argument('-l','--list_dir', type=str, default=LIST_PATH,
					help='directory to output list')	
args = parser.parse_args()

data_root = Path(args.data_root)
list_dir = Path(args.list_dir)

train_dir = list_dir / 'vctk_train'
train_dir.mkdir(parents=True, exist_ok=True)

test_dir = list_dir / 'vctk_test'
test_dir.mkdir(parents=True, exist_ok=True)

wf_train_wavscp = open(str(train_dir/'wav.scp'),'w')
wf_train_utt2spk = open(str(train_dir/'utt2spk'),'w')
wf_test_wavscp = open(str(test_dir/'wav.scp'),'w')
wf_test_utt2spk = open(str(test_dir/'utt2spk'),'w')

for speaker_name in sorted(list(data_root.glob('*'))):
	for data_file in sorted(list(speaker_name.glob('*.wav'))):
		data_num = int(data_file.stem.split('_')[-1])
		if data_num <= 20:
			wf_test_wavscp.write('{} {}\n'.format(data_file.stem,str(data_file.absolute())))
			wf_test_utt2spk.write('{} {}\n'.format(data_file.stem,speaker_name.stem))
		else:
			wf_train_wavscp.write('{} {}\n'.format(data_file.stem,str(data_file.absolute())))
			wf_train_utt2spk.write('{} {}\n'.format(data_file.stem,speaker_name.stem))

wf_train_wavscp.close()
wf_train_utt2spk.close()
wf_test_wavscp.close()
wf_test_utt2spk.close()
