import os
import time
from pathlib import Path
from importlib import import_module
from scipy.io import wavfile

import numpy as np
import librosa
from librosa.filters import mel as librosa_mel_fn

import torch
import torch.nn.functional as F

from util.stft import STFT
from util.mel_spectrum import MelSpectrum
from util.wld_vocoder import Wld_vocoder
from util.wavlm_stat_norm import WavLM_function
from parallel_wavegan.utils import write_hdf5
import ipdb
import yaml

MAX_WAV_VALUE = 32768.0
# PWG_MODEL_PATH = 'baseline_pwg_model/checkpoint-400000steps.pkl'
# PWG_STATS_PATH = 'baseline_pwg_model/stats.h5'

# PWG_MODEL_PATH = 'downloads/pwg_mel-wavlm_400-320_18tmsv/checkpoint-500000steps.pkl'
# PWG_STATS_PATH = 'downloads/pwg_mel-wavlm_400-320_18tmsv/stats.h5'
# PWG_MODEL_PATH = 'downloads/pwg_mel-wavlm_400-320/checkpoint-500000steps.pkl'
# PWG_STATS_PATH = 'downloads/pwg_mel-wavlm_400-320/stats.h5'
# PWG_MODEL_PATH = 'downloads/pwg_concate_feat_18tmsv/checkpoint-500000steps.pkl'
# PWG_STATS_PATH = 'downloads/pwg_concate_feat_18tmsv/stats.h5'
PWG_MODEL_PATH = 'downloads/pwg_wavlm_18tmsv_new/checkpoint-500000steps.pkl'
PWG_STATS_PATH = 'downloads/pwg_wavlm_18tmsv_new/stats.h5'
# PWG_MODEL_PATH = 'downloads/pwg_NL01/checkpoint-400000steps.pkl'
# PWG_STATS_PATH = 'downloads/pwg_NL01/stats.h5'
# PWG_MODEL_PATH = 'downloads/apmcc_pwg_NL01/checkpoint-500000steps.pkl'
# PWG_STATS_PATH = 'downloads/apmcc_pwg_NL01/stats.h5'
# PWG_MODEL_PATH = '/home/bioasp/Downloads/pwg_NL01v2_mccap/checkpoint-600000steps.pkl'
# PWG_STATS_PATH = '/home/bioasp/Downloads/pwg_NL01v2_mccap/stats.h5'

# HIFI_MODEL_PATH = 'downloads/hifi_GAN/checkpoint-1400000steps.pkl'
# HIFI_STATS_PATH = 'downloads/hifi_GAN/stats.h5'
# HIFI_CONFIG_PATH = 'downloads/hifi_GAN/config.yml'
# PWG_MODEL_PATH = 'downloads/hifi_GAN/checkpoint-1400000steps.pkl'
# PWG_STATS_PATH = 'downloads/hifi_GAN/stats.h5'

use_pwg = True


import h5py
def read_hdf5(hdf5_name, hdf5_path):
    """Read hdf5 dataset.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Return:
        any: Dataset values.

    """
    if not os.path.exists(hdf5_name):
        print(f"There is no such a hdf5 file ({hdf5_name}).")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        print(f"There is no such a data in hdf5 file. ({hdf5_path})")
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data


def load_pwg():
    pwg_config = {
        "generator_params":{
            "aux_channels": 768,
            "aux_context_window": 2,
            "dropout": 0.0,
            "gate_channels": 128,
            "in_channels": 1,
            "kernel_size": 3,
            "layers": 30,
            "out_channels": 1,
            "residual_channels": 64,
            "skip_channels": 64,
            "stacks": 3,
            "upsample_net": "ConvInUpsampleNetwork",
            "upsample_params":{
                "upsample_scales": [4, 4, 4, 5]
                # "upsample_scales": [4, 5, 3, 5]
                # "upsample_scales": [4, 4, 4, 4]

            },
            "use_weight_norm": True,
        }
    }
    from model.parallelwavegan import ParallelWaveGANGenerator
    pwg_model = ParallelWaveGANGenerator(**pwg_config["generator_params"])
    pwg_model.load_state_dict(
        torch.load(PWG_MODEL_PATH, map_location="cpu")["model"]["generator"])
    pwg_model.remove_weight_norm()

    mean_ = read_hdf5(PWG_STATS_PATH, "mean")
    scale_ = read_hdf5(PWG_STATS_PATH, "scale")
    mean_ = torch.from_numpy(mean_).float().cuda().view(1,-1,1)
    scale_ = torch.from_numpy(scale_).float().cuda().view(1,-1,1)

    return pwg_model.eval().cuda(), {'mean':mean_,'scale':scale_}
    

def inference(trials, utt2path, spk2spk_id,
              model_path,
              output_mcc_dir, output_wav_dir,
              model_type):

    stat_dict       = data_config.get('statistic_file', None)
    feat_kind       = data_config.get('feature_kind', 'mel')
    fft_size        = data_config.get('fft_size', 1024)
    shiftms         = data_config.get('shiftms', 5)
    hop_length      = data_config.get('hop_length', 256)
    win_length      = data_config.get('win_length', 1024)
    n_mel_channels  = data_config.get('n_mel_channels', 80)
    mcc_dim         = data_config.get('mcc_dim', 24)
    mcc_alpha       = data_config.get('mcc_alpha', 0.455)    
    sampling_rate   = data_config.get('sampling_rate', 24000)
    mel_fmin        = data_config.get('mel_fmin', 80)
    mel_fmax        = data_config.get('mel_fmax', 7600)
    f0_fmin         = data_config.get('f0_fmin', 80)
    f0_fmax         = data_config.get('f0_fmax', 7600)
    # f0_fmin         = data_config.get('f0_fmin', 40)
    # f0_fmax         = data_config.get('f0_fmax', 700)
    cutoff_freq     = data_config.get('cutoff_freq', 70)


    module = import_module('model.{}'.format(model_type), package=None)
    MODEL = getattr(module, 'Model')
    model = MODEL(model_config)

    model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
    model.cuda().eval()

    feat_kinds = feat_kind.split('-')

    # ## hifi GAN
    # with open(HIFI_CONFIG_PATH) as f:
    #     config = yaml.load(f, Loader=yaml.Loader)
    # config.update(vars(args))
    ## ap_mcc pwg
    if use_pwg and feat_kinds[-1] == 'WavLM':
        pwg_model, pwg_stats = load_pwg()
        print("nice WavLM")

    ## ap_mcc pwg
    if use_pwg and feat_kinds[-1] == 'mcc':
        pwg_model, pwg_stats = load_pwg()
        print("nice mcc")

    ## mel_pwg
    if use_pwg and feat_kinds[-1] in ['mel','stft']:
        pwg_model, pwg_stats = load_pwg()
        print("nice mel pwg")


    if stat_dict is not None:
        feat_stat = torch.load(stat_dict)
        print(stat_dict)
        print(" hav ")
        # ipdb.set_trace()
    else:
        feat_stat = None
        # ipdb.set_trace()
        print(" None ")

    if 'stft' in feat_kinds:
        stft_fn = STFT(filter_length=fft_size,
                       hop_length=hop_length, 
                       win_length=win_length, 
                       window='hann',
                       feat_stat=feat_stat).cuda()

    if 'mel' in feat_kinds:
        mel_fn = MelSpectrum(filter_length=fft_size,
                             hop_length=hop_length,
                             win_length=win_length,
                             n_mel_channels=n_mel_channels,
                             sampling_rate=sampling_rate,
                             mel_fmin=mel_fmin, mel_fmax=mel_fmax,
                             feat_stat=feat_stat).cuda()
    if 'WavLM' in feat_kinds:
        WavLM_fn = WavLM_function

    if sum([(f in feat_kinds) for f in ['sp','mcc','f0','ap','wld']]) > 0:
        wld_fn = Wld_vocoder(fft_size=fft_size, 
                             shiftms=shiftms, 
                             sampling_rate=sampling_rate, 
                             mcc_dim=mcc_dim, mcc_alpha=mcc_alpha, 
                             minf0=f0_fmin, maxf0=f0_fmax,
                             cutoff_freq=cutoff_freq,
                             feat_stat=feat_stat).cuda()

    for i, trial in enumerate(trials):
        print(trial)
        if len(trial) == 3:
            source_data, source_name, target_name = trial
            print(source_data) ## source == NL01_241
            target_data = None
        else:
            source_data, source_name, target_data, target_name = trial

        # Load source data
        if utt2path['type'] == 'feats':
            X = torch.load(utt2path[source_data])
            X = dict([[key,val.cuda().unsqueeze(0)] for key,val in X.items()])
            for key in X:
                print(key , X[key].shape) ## key == mel sp mcc ap ....
            X_in = [X[feat_kind].clone() for feat_kind in feat_kinds[:-1]]
           
            # X_in = [X["dtw_el_mel"].clone()]

            # print("feat_kind")
            # for aaaa in feat_kinds[:-1] :
            #     print(aaaa)
            # print("feat_kind")
            # X_in_ap = X["ap"].clone()
            
            # print("x_in")
            # print(X_in)
            # print(X_in_ap)

        elif utt2path['type'] == 'wav':
            fs, X = wavfile.read(utt2path[source_data])
            assert fs == sampling_rate
            X = X / MAX_WAV_VALUE

            X = torch.from_numpy(X).float().cuda().unsqueeze(0)

            # Convert spectral features
            if feat_kind.split('-')[0] == 'wav':
                X_in = X.unsqueeze(1)
                X = {'wav':X}

            elif feat_kind.split('-')[0] == 'stft':
                X = stft_fn(X)
                X_in = stft_fn.normalize(X)['spec']

            elif feat_kind.split('-')[0] == 'mel':
                X = mel_fn(X)
                X_in = mel_fn.normalize(X)['mel']

            else :
                X = wld_fn(X)
                X_in = wld_fn.normalize(X)[feat_kinds[0]]


        Y_in = torch.ones(1,1) * spk2spk_id[target_name]
        Y_in = Y_in.long().cuda()
        X_in.append(Y_in)


        with torch.no_grad():
            Y_out = model(X_in, encoder_kind= str(feat_kind.split('-')[-1]), 
                                decoder_kind= str(feat_kind.split('-')[-1]))

        print("type(Y_out)") ## OUTCOME TENSOR
        print(type(Y_out))
        print(Y_out.shape)
        # ipdb.set_trace()
        # De-normalize
        if feat_kind.split('-')[-1] == 'wav':
            Y_out_dn = Y_out

        elif feat_kind.split('-')[-1] == 'stft':
            Y_out_dn = stft_fn.denormalize({'spec':Y_out})['spec']

        elif feat_kind.split('-')[-1] == 'mel':
            Y_out_dn = mel_fn.denormalize({'mel':Y_out})['mel']

        elif feat_kind.split('-')[-1] == 'WavLM':
            Y_out_dn = WavLM_fn.denormalize({'WavLM':Y_out})['WavLM']
        else:
            Y_out_dn = dict([[key,val.clone()] for key,val in X.items()])
            Y_out_dn[feat_kinds[-1]] = Y_out
            X = wld_fn.denormalize(X)
            Y_out_dn = wld_fn.denormalize(Y_out_dn)
            

        # NORMALIZE 2        
        # got WavLM_feat 
        # converted_WavLM = Y_out_dn.cpu().detach().numpy()
        # assert (converted_WavLM.shape[0]==768) | (converted_WavLM.shape[1]==768) 
        
        # source_number = source_data.split('_')[-1]
        # filepath_WavLM = os.path.join("converted_WavLM/", f"EL01_{source_number}.npy")
        # np.save(filepath_WavLM,converted_WavLM)

        concate_feat_flag = False

        if concate_feat_flag == True :
            source_number = source_data.split('_')[-1]
            dir_path = "/home/4TB_storage/hsinhao_storage/AVEL_align/cdvae-align-feature-sly-align/converted_WavLM/"
            data_path = "EL01_" + str(source_number) +".npy"
            filepath_np = os.path.join(dir_path, data_path)
            WavLM_feat = np.load(filepath_np)
            mel = torch.log10(torch.clamp(Y_out_dn, min=1e-10))
            mel = mel.cpu().detach().numpy()

            print(WavLM_feat.shape)
            print(mel.shape)

            already_dtw = True
            if already_dtw == True :
                assert (WavLM_feat.shape[2] == mel.shape[2])
            else :
                if WavLM_feat.shape[2] == (mel.shape[2] - 2):
                    mel = mel[:,:,1:-1]
                else:
                    mel = mel[:,:,0:-1]
                print(mel.shape)

            ## for already dtw WavLM and mel
            assert (WavLM_feat.shape[2] == mel.shape[2])

            concate_WavMel = np.concatenate((WavLM_feat,mel),axis = 1)
            print(concate_WavMel.shape)
            
            assert (concate_WavMel.shape[0]==848) | (concate_WavMel.shape[1]==848)            
            filepath_WavMel = os.path.join("converted_concate_WavMel/", f"EL01_{source_number}.npy")
            np.save(filepath_WavMel,concate_WavMel)
            
            concate_feat = torch.tensor(concate_WavMel).cuda()
        # ipdb.set_trace()

        ### record (25 dim)mcc for mcc_plot
        # filepath_mcc = os.path.join("/home/bioasp/Desktop/mcc_np/stage2_converted/", f"NL01_{source_number}.npy")
        # np.save(filepath_mcc,mcc_np)   

        # for i in Y_out_dn :
        #     print(i , Y_out_dn[i].shape)
        # ipdb.set_trace()
        # Synthesis
        hop_frame_size = 320
        if concate_feat_flag == True :
            with torch.no_grad():
                print("using concate pwg")
                # Y_out_rn = torch.log10(torch.clamp(Y_out_dn, min=1e-10))
                Y_out_rn = (concate_feat - pwg_stats['mean']) / pwg_stats['scale']
                # Y_out_rn = concate_feat
                Z_in = torch.randn(1, 1, Y_out_rn.shape[2] * 320, device=Y_out_rn.device)
                Y_out_rn = F.pad(Y_out_rn,(2,2),mode='replicate')
                yhat = pwg_model(Z_in, Y_out_rn).view(-1)   

            yhat = yhat.cpu().numpy()

        elif feat_kinds[-1] == 'wav':
            yhat = Y_out_dn.cpu().numpy()[0,0]

        elif feat_kinds[-1] == 'stft' and use_pwg:
            with torch.no_grad():
                Y_out_rn = torch.matmul(mel_basis, Y_out_dn)
                Y_out_rn = torch.log10(torch.clamp(Y_out_rn, min=1e-10))
                Y_out_rn = (Y_out_rn - pwg_stats['mean']) / pwg_stats['scale']
                Z_in = torch.randn(1, 1, Y_out_rn.shape[2] * 256, device=Y_out_rn.device)
                Y_out_rn = F.pad(Y_out_rn,(2,2),mode='replicate')
                yhat = pwg_model(Z_in, Y_out_rn).view(-1)    
            yhat = yhat[0].cpu().numpy()

        elif feat_kinds[-1] == 'stft':
            Y_out_np = Y_out_dn.cpu().numpy()[0]
            yhat = librosa.core.griffinlim(Y_out_np,  
                                n_iter=150, 
                                hop_length=256, 
                                win_length=1024, 
                                window='hann')

        elif feat_kinds[-1] == 'mel' and use_pwg:
            with torch.no_grad():
                print("using mel pwg")
                Y_out_rn = torch.log10(torch.clamp(Y_out_dn, min=1e-10))
                Y_out_rn = (Y_out_rn - pwg_stats['mean']) / pwg_stats['scale']
                Z_in = torch.randn(1, 1, Y_out_rn.shape[2] * hop_frame_size, device=Y_out_rn.device)
                Y_out_rn = F.pad(Y_out_rn,(2,2),mode='replicate')
                yhat = pwg_model(Z_in, Y_out_rn).view(-1)   

            yhat = yhat.cpu().numpy()

        elif feat_kinds[-1] == 'WavLM' and use_pwg:
            with torch.no_grad():
                print("using WavLM pwg")
                # Y_out_rn = torch.log10(torch.clamp(Y_out_dn, min=1e-10))
                Y_out_rn = (Y_out_dn - pwg_stats['mean']) / pwg_stats['scale']
                # Y_out_rn = Y_out_dn
                Z_in = torch.randn(1, 1, Y_out_rn.shape[2] * hop_frame_size, device=Y_out_rn.device)
                Y_out_rn = F.pad(Y_out_rn,(2,2),mode='replicate')
                yhat = pwg_model(Z_in, Y_out_rn).view(-1)   

            yhat = yhat.cpu().numpy()

        # X_in_ap = X["ap"].clone()
        elif feat_kinds[-1] == 'mcc' and use_pwg:
            with torch.no_grad():
                ### record
                mcc_feat = Y_out_dn["mcc"][:,1:]
                X_in_ap = Y_out_dn["ap"]
                Y_out_rn = torch.cat((mcc_feat,X_in_ap),1)

                Y_out_rn = (Y_out_rn - pwg_stats['mean']) / pwg_stats['scale']
                Z_in = torch.randn(1, 1, Y_out_rn.shape[2] * 256, device=Y_out_rn.device)
                Y_out_rn = F.pad(Y_out_rn,(2,2),mode='replicate')
                print("using apmcc-PWG")
                
                yhat = pwg_model(Z_in, Y_out_rn).view(-1)   

            yhat = yhat.cpu().numpy()

        elif feat_kinds[-1] == 'mel':
            Y_out_np = Y_out_dn.cpu().numpy()[0]
            yhat = librosa.feature.inverse.mel_to_audio(Y_out_np, 
                                sr=fs, 
                                n_fft=filter_length, 
                                hop_length=hop_length, 
                                win_length=win_length, 
                                window='hann', 
                                center=True,
                                pad_mode='reflect', 
                                power=1.0, 
                                n_iter=32,
                                fmin=mel_fmin, fmax=mel_fmax)
        else :
            # yhat = wld_fn.synthesis(X, se_kind=feat_kinds[-1]).view(-1)
            yhat = wld_fn.synthesis(Y_out_dn, se_kind=feat_kinds[-1]).view(-1)
            yhat = yhat.cpu().numpy()
            print(Y_out_dn["f0"])
            print("now using WLD vocoder")


        data_name = '_'.join(source_data.split('_')[1:])

        # Save converted feats
        if output_mcc_dir is not None:
            feat_path = output_feat_dir / "{}_{}_{}.pt".format(source_name,target_name,data_name)
            torch.save(  Y_out_dn, feat_path)
        
        # Save converted WAV
        if output_wav_dir is not None:
            audio_path = output_wav_dir / "{}_{}_{}.wav".format(source_name,target_name,data_name)
            yhat = yhat * MAX_WAV_VALUE
            wavfile.write( audio_path, sampling_rate, yhat.astype('int16'))

        print('Generate {}.'.format( audio_path),end='\n')
    # print('Generate {}.'.format( audio_path),end='\n')


if __name__ == "__main__":
    import argparse
    import json
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/config_vae_vctk.json',
                        help='JSON file for configuration')
    parser.add_argument('-d','--data_dir', type=str, default=None,
                        help='Dir. for input data')
    parser.add_argument('-t','--trials', type=str, default=None,
                        help='Trials file for conversion')
    parser.add_argument('-m','--model_path', type=str, default=None,
                        help='Path to checkpoint with model')
    parser.add_argument('-S','--statistic_file', type=str, default=None,
                        help='Statistic file')
    parser.add_argument('-K','--feature_kind', type=str, default=None,
                        help='Feature kinds')
    parser.add_argument('-om', "--output_mcc_dir", type=str, default=None)
    parser.add_argument('-ow', "--output_wav_dir", type=str, default=None)    
    parser.add_argument('-g', "--gpu", type=str, default='0,1')

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load Config.
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    infer_config = config["infer_config"] 
    global data_config
    data_config = config["data_config"]
    if args.feature_kind is not None:
        data_config['feature_kind'] = args.feature_kind
    if args.statistic_file is not None:
        data_config['statistic_file'] = args.statistic_file

    if args.model_path is None:
        model_path = infer_config["model_path"]
    else:
        model_path = args.model_path

    model_type = infer_config["model_type"]    

    global model_config
    # Fix model config if GAN is used
    if 'Generator' in config["model_config"].keys():
        model_config = config["model_config"]['Generator']
    else:
        model_config = config["model_config"]
    # print("config_model_config")
    # print(config["model_config"])
    # ipdb.set_trace()

    # Load wav.scp & spk2spk_id
    if args.data_dir is None:
        data_dir = Path(data_config["testing_dir"])
    else:
        data_dir = Path(args.data_dir)
    if (data_dir / 'feats.scp').exists():
        with open(data_dir / 'feats.scp','r') as rf:
            utt2path = dict([line.rstrip().split() for line in rf.readlines()])
            utt2path['type'] = 'feats'
    else:
        with open(data_dir / 'wav.scp','r') as rf:
            utt2path = dict([line.rstrip().split() for line in rf.readlines()])
            utt2path['type'] = 'wav'
    with open(data_dir / 'spk2spk_id','r') as rf:
        spk2spk_id = [line.rstrip().split() for line in rf.readlines()]
        spk2spk_id = dict([[spk, int(spk_id)] for spk, spk_id in spk2spk_id])

    # Load trials
    if args.trials is None:
        trials_path = infer_config["trials"]
    else:
        trials_path = args.trials
    with open(trials_path,'r') as rf:
        trials = [line.rstrip().split() for line in rf.readlines()]

    # Make dir. for outputing converted feature
    if args.output_mcc_dir is None:
        output_mcc_dir = infer_config["output_mcc_dir"]
    else:
        output_mcc_dir = args.output_mcc_dir
    if output_mcc_dir is not None:
        output_mcc_dir = Path(output_mcc_dir)
        output_mcc_dir.mkdir(parents=True, exist_ok=True)

    # Make dir. for outputing converted waveform
    if args.output_wav_dir is None:
        output_wav_dir = infer_config["output_wav_dir"]
    else:
        output_wav_dir = args.output_wav_dir
    if output_wav_dir is not None:
        output_wav_dir = Path(output_wav_dir)
        output_wav_dir.mkdir(parents=True, exist_ok=True)        

    inference(trials, utt2path, spk2spk_id,
              model_path,
              output_mcc_dir, output_wav_dir,
              model_type)
