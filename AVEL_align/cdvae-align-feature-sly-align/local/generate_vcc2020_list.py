import os
import ipdb
from pathlib import Path
from scipy.io import wavfile


ROOT_PATH = '/mnt/md0/user_roland/vcc2020_training'
LIST_PATH = 'data'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d','--data_root', type=str, default=ROOT_PATH,
					help='Corpus path')	
parser.add_argument('-l','--list_dir', type=str, default=LIST_PATH,
					help='directory to output list')	
args = parser.parse_args()


speaker_list = ['SEF1','SEF2','SEM1','SEM2','TEF1','TEF2','TEM1','TEM2']

data_root = Path(args.data_root)
list_dir = Path(args.list_dir)

train_dir = list_dir / 'vcc2020_train'
train_dir.mkdir(parents=True, exist_ok=True)

test_dir = list_dir / 'vcc2020_test'
test_dir.mkdir(parents=True, exist_ok=True)

wf_train_wavscp = open(str(train_dir/'wav.scp'),'w')
wf_train_utt2spk = open(str(train_dir/'utt2spk'),'w')
wf_test_wavscp = open(str(test_dir/'wav.scp'),'w')
wf_test_utt2spk = open(str(test_dir/'utt2spk'),'w')

for speaker_name in speaker_list:
	for data_file in sorted(list((data_root/speaker_name).glob('*.wav'))):
		data_num = int(data_file.stem[-2:])
		if data_num >= 51 and data_num <= 70:
			wf_test_wavscp.write('{}_{} {}\n'.format(speaker_name,data_file.stem,str(data_file.absolute())))
			wf_test_utt2spk.write('{}_{} {}\n'.format(speaker_name,data_file.stem,speaker_name))
		else:
			wf_train_wavscp.write('{}_{} {}\n'.format(speaker_name,data_file.stem,str(data_file.absolute())))
			wf_train_utt2spk.write('{}_{} {}\n'.format(speaker_name,data_file.stem,speaker_name))

wf_train_wavscp.close()
wf_train_utt2spk.close()
wf_test_wavscp.close()
wf_test_utt2spk.close()

