import os
import ipdb
from pathlib import Path
from scipy.io import wavfile


ROOT_PATH = '/mnt/md0/user_ymchiqq/dataset/ELVC/EL01'
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
speaker_name=Path(data_root).name

print(f'root path: {data_root}')
print(args.list_dir)
print(f'data dir: {list_dir}')
print(f'speaker_name: {speaker_name}')

train_dir = list_dir / 'tmhint_el_train'
train_dir.mkdir(parents=True, exist_ok=True)

test_dir = list_dir / 'tmhint_el_test'
test_dir.mkdir(parents=True, exist_ok=True)

wf_train_wavscp = open(str(train_dir/'wav.scp'),'w')
wf_train_utt2spk = open(str(train_dir/'utt2spk'),'w')
wf_test_wavscp = open(str(test_dir/'wav.scp'),'w')
wf_test_utt2spk = open(str(test_dir/'utt2spk'),'w')


for data_file in sorted(list(data_root.glob('*.wav'))):
	# sp: EL01
	# utt: EL01_001
	
	cur_idx = int(data_file.stem.split('_')[1])
	cur_utt =data_file.stem
	cur_sp = data_file.stem.split('_')[0]
	
	if cur_idx > 240:
		wf_test_wavscp.write('{} {}\n'.format(cur_utt, str(data_file.absolute())))
		wf_test_utt2spk.write('{} {}\n'.format(cur_utt, cur_sp))
	else:
		wf_train_wavscp.write('{} {}\n'.format(cur_utt, str(data_file.absolute())))
		wf_train_utt2spk.write('{} {}\n'.format(cur_utt, cur_sp))

wf_train_wavscp.close()
wf_train_utt2spk.close()
wf_test_wavscp.close()
wf_test_utt2spk.close()
