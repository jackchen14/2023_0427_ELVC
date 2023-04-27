from pathlib import Path
import os
from os import walk
from os.path import basename, dirname, join, exists, splitext

def file2list(list_path):
    list_data = []
    with open(list_path, 'r', encoding='utf-8') as fid:

        for line_idx, line in enumerate(fid):
            fpath = line.rstrip()
            list_data.append(fpath)

    return list_data

def list2file(file_list, list_data):
    fid = open(file_list, 'r')
    for item in list_data:
        fid.write(item)
    fid.close()

def path2list(filepath, tar_ext='wav'):
    list_all = []
    for dirPath, dirNames, fileNames in os.walk(filepath):
        for n_itr, f in enumerate(fileNames):            
            cur_ext = Path(f).suffix[1:]
            if cur_ext == tar_ext:
                list_all.append(Path(dirPath, f))
    return list_all

def gen_esplist(list_ts, list_wav, list_idx, data_dir, sp_id, task_id):

    list_wav.sort()
    
    if len(task_id)==0:
        output_dir = Path(data_dir, f'{sp_id}')
    else:
        output_dir = Path(data_dir, f'{sp_id}_{task_id}')
        
    os.makedirs(output_dir, exist_ok=True)
    
    ### write wav.scp
    file_wavscp = Path(output_dir, 'wav.scp')
    fid = open(file_wavscp, 'w')
    for idx in list_idx:
        fp = list_wav[idx]
        fname = Path(list_wav[idx]).stem
        fid.write(f'{fname} {fp}\n')
    fid.close()
    
    ### write utt2spk
    file_utt2spk = Path(output_dir, 'utt2spk')
    fid = open(file_utt2spk, 'w')
    for idx in list_idx:
        fname = Path(list_wav[idx]).stem
        fid.write(f'{fname} {sp_id}\n')
    fid.close()
    
    ### write test
    file_text = Path(output_dir, 'text')
    fid = open(file_text, 'w', encoding='utf-8')
    for idx in list_idx:
        fname = Path(list_wav[idx]).stem
        fid.write(f'{fname} {list_ts[idx]}\n')
    fid.close()
    
    ### write spk2utt
    file_spk2utt = Path(output_dir, 'spk2utt')
    fid = open(file_spk2utt, 'w', encoding='utf-8')
    fid.write(f'{sp_id}')
    for idx in list_idx:
        fname = Path(list_wav[idx]).stem
        fid.write(f' {fname}')
    fid.write(f'\n')
    fid.close()


def pwg_data(args, task_id):

    list_wav_org = path2list(args.input_dir)
    
    list_wav = sorted(list_wav_org)
    dirname = Path(args.input_dir).name
    sp_id = dirname
    # print(list_wav)
    # print(len(list_wav))

    output_dir = Path(args.data_dir, f'{sp_id}_{task_id}')
    os.makedirs(output_dir, exist_ok=True)


    file_wavscp = Path(output_dir, 'wav.scp')
    fid = open(file_wavscp, 'w')

    train_num = args.n_train
    dev_num = args.n_train + args.n_dev
    eval_num = args.n_train + args.n_dev + args.n_eval
    if task_id == 'train':
        print("-------train_set--------")
        for data_path in list_wav :
            cur_idx = int(data_path.stem.split('_')[1])
            if cur_idx <= train_num :
                fp = data_path
                fname = Path(data_path).stem
                print(fname)
                fid.write(f'{fname} {fp}\n')
    if task_id == 'dev':
        print("-------dev_set--------")
        for data_path in list_wav :
            cur_idx = int(data_path.stem.split('_')[1])
            if  train_num < cur_idx <= dev_num :
                fp = data_path
                fname = Path(data_path).stem
                print(fname)
                fid.write(f'{fname} {fp}\n')
    if task_id == 'eval':
        print("-------eval_set--------")
        for data_path in list_wav :
            cur_idx = int(data_path.stem.split('_')[1])
            if  dev_num < cur_idx <= eval_num :
                fp = data_path
                fname = Path(data_path).stem
                print(fname)
                fid.write(f'{fname} {fp}\n')


    
    fid.close()
    print(f'wav.scp is saved at: {file_wavscp}')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default='/mnt/md2/datasets/EL/ELVC/NL01', help='input wav dir')
    parser.add_argument('-nt', '--n_train', type=int, default=280, help='samples for trian')
    parser.add_argument('-ne', '--n_eval', type=int, default=20, help='samples for eval')
    parser.add_argument('-nd', '--n_dev', type=int, default=20, help='samples for dev')
    parser.add_argument('-d', '--data_dir', type=str, default='./data', help='path to audio')    
    args = parser.parse_args()

    
    print(f'data dir: {args.input_dir}')
    print(f'samples of training:')
    print(f'train: {args.n_train}')
    print(f'dev: {args.n_dev}')
    print(f'eval: {args.n_eval}')

    pwg_data(args, task_id='train')
    pwg_data(args, task_id='dev')
    pwg_data(args, task_id='eval')
