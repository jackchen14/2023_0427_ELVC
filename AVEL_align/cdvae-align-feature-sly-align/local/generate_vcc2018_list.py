'''
	list format:
		utt_id_1 wav_path_1
		utt_id_2 wav_path_2
		...
		utt_id_N wav_path_N
'''
import os
from pathlib import Path

def main(source_name, data_root, list_dir, train_ndata, test_ndata):
	# List for feature extraction
	source_list_file = list_dir / '{}_train.list'.format(source_name)
	with open(str(source_list_file),'w') as wf:
		for n in range(train_ndata):
			wf.write('{0}_100{1:02d} {2}/tr/{0}/200{1:02d}.wav\n'.format(source_name,n+1,data_root))

	source_list_file = list_dir / '{}_test.list'.format(source_name)
	with open(str(source_list_file),'w') as wf:
		for n in range(test_ndata):
			wf.write('{0}_300{1:02d} {2}/ev/{0}/300{1:02d}.wav\n'.format(source_name,n+1,data_root))

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('source_name', type=str,
						help='source speaker name')
	parser.add_argument('data_root', type=str,
						help='Corpus path')	
	parser.add_argument('-d','--list_dir', type=str, default='data/list',
						help='directory to output list')	
	parser.add_argument('-n','--train_ndata', type=int, default=81,
						help='number of training data')
	parser.add_argument('-m','--test_ndata', type=int, default=35,
						help='number of testing data')
	args = parser.parse_args()

	list_dir = Path(args.list_dir)
	list_dir.mkdir(parents=True, exist_ok=True)
	main( args.source_name, args.data_root, list_dir, args.train_ndata, args.test_ndata)