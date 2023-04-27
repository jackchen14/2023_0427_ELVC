import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
from util.wld_vocoder import Wld_vocoder
from util.mel_spectrum import MelSpectrum
from librosa.display import specshow
from pathlib import Path
from scipy.io import wavfile
import os
import torch.nn.functional as F
import ipdb
import h5py
import pickle


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

# stat_dict       = data_config.get('statistic_file', None)
# feat_kind       = data_config.get('feature_kind', 'mel')
# fft_size        = data_config.get('fft_size', 1024)
# shiftms         = data_config.get('shiftms', 5)
# hop_length      = data_config.get('hop_length', 256)
# win_length      = data_config.get('win_length', 1024)
# n_mel_channels  = data_config.get('n_mel_channels', 80)
# mcc_dim         = data_config.get('mcc_dim', 24)
# mcc_alpha       = data_config.get('mcc_alpha', 0.455)    
# sampling_rate   = data_config.get('sampling_rate', 24000)
# mel_fmin        = data_config.get('mel_fmin', 80)
# mel_fmax        = data_config.get('mel_fmax', 7600)
# f0_fmin         = data_config.get('f0_fmin', 80)
# f0_fmax         = data_config.get('f0_fmax', 7600)
# # f0_fmin         = data_config.get('f0_fmin', 40)
# # f0_fmax         = data_config.get('f0_fmax', 700)
# cutoff_freq     = data_config.get('cutoff_freq', 70)
# # feat_stat = torch.load("data/train_nl_cmvn/stats.pt")

# mel_fn = MelSpectrum(filter_length=fft_size,
#                         hop_length=hop_length,
#                         win_length=win_length,
#                         n_mel_channels=n_mel_channels,
#                         sampling_rate=sampling_rate,
#                         mel_fmin=mel_fmin, mel_fmax=mel_fmax,
#                         feat_stat=feat_stat).cuda()
# load feature
def feature_to_audio():
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
    feat_stat = torch.load(stat_dict)

    mel_fn = MelSpectrum(filter_length=fft_size,
                        hop_length=hop_length,
                        win_length=win_length,
                        n_mel_channels=n_mel_channels,
                        sampling_rate=sampling_rate,
                        mel_fmin=mel_fmin, mel_fmax=mel_fmax,
                        feat_stat=feat_stat).cuda()
    # f0_fmin         = data_config.get('f0_fmin', 40)
    # f0_fmax         = data_config.get('f0_fmax', 700)
    cutoff_freq     = data_config.get('cutoff_freq', 70)
    # output_wav_type = "nl"
    # output_wav_type_big = "NL"

    output_wav_type = "el"
    output_wav_type_big = "EL"
    for data_num in range(240) :
        output_file_name = "NL01_" + "%03d" % (data_num+1)
        if output_wav_type == "nl" :
            feat_path = "/home/4TB_storage/hsinhao_storage/AVEL_align/cdvae-align-feature-sly-align/features/tmhint_" + output_wav_type + "_train/" + output_file_name + ".pt"
        else :
            feat_path = "/home/4TB_storage/hsinhao_storage/AVEL_align/cdvae-align-feature-sly-align/features/tmhint_" + output_wav_type + "_train/" + "EL01_" + output_file_name.split("_")[-1] + ".pt"
        feats = torch.load(feat_path)
        print("org_feats")
        for feat_name in feats :
            print(feat_name , feats[feat_name].shape)


        ## load aligned index
        if output_wav_type == "nl" :
            sly_align_data = np.load("./sly_aligned_NL_array/" + output_file_name + ".npy")
        else :
            sly_align_data = np.load("./sly_aligned_EL_array/" + output_file_name + ".npy")

        ## insert aligned index into feature 
        temp_feats = {}
        for i in feats :
            if i == 'WavLM' :
                temp = np.empty([len(feats[i][:,0]),len(sly_align_data)])
                count = 0
                for j in sly_align_data :
                    temp[:,count] = feats[i][:,j]
                    count += 1
                temp_feats[i] = torch.from_numpy(temp).float()
                temp_feats[i] = temp_feats[i].reshape(1,temp_feats[i].shape[0],temp_feats[i].shape[1])
            elif i == 'f0' :
                temp = np.empty([len(sly_align_data)])
                count = 0
                for j in sly_align_data :
                    temp[count] = feats[i][j]
                    count += 1
                temp_feats[i] = torch.from_numpy(temp).float()
                temp_feats[i] = temp_feats[i].reshape(1,temp_feats[i].shape[0])
            else :
                temp = np.empty([len(feats[i][:,0]),len(sly_align_data)])
                count = 0
                for j in sly_align_data :
                    temp[:,count] = feats[i][:,j]
                    count += 1
                temp_feats[i] = torch.from_numpy(temp).float()
                temp_feats[i] = temp_feats[i].reshape(1,temp_feats[i].shape[0],temp_feats[i].shape[1])


        print("temp_feats")
        for i in temp_feats :
            print(i , temp_feats[i].shape)
                                
        ### synthesize
        ### PWG
        PWG_MODEL_PATH = 'downloads/pwg_mel-wavlm_400-320/checkpoint-500000steps.pkl'
        PWG_STATS_PATH = 'downloads/pwg_mel-wavlm_400-320/stats.h5'

        def load_pwg():
            pwg_config = {
                "generator_params":{
                    "aux_channels": 80,
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
        
        use_pwg = True
        if use_pwg == True :
            pwg_model, pwg_stats = load_pwg()
            ### mcc ap vocoder ###
            # with torch.no_grad():
            #     ### record
            #     mcc_feat = temp_feats["mcc"][:,1:]
            #     X_in_ap = temp_feats["ap"]
            #     Y_out_rn = torch.cat((mcc_feat,X_in_ap),1).cuda()

            #     Y_out_rn = (Y_out_rn - pwg_stats['mean']) / pwg_stats['scale']
            #     Z_in = torch.randn(1, 1, Y_out_rn.shape[2] * 256, device=Y_out_rn.device)
            #     Y_out_rn = F.pad(Y_out_rn,(2,2),mode='replicate')
            #     print("using apmcc-PWG")           
            #     yhat = pwg_model(Z_in, Y_out_rn).view(-1)
            Y_out_dn = temp_feats["mel"]
            # Y_out_dn = mel_fn.denormalize({'mel':Y_out_dn.cuda()})['mel']
            ### mel vocoder ###
            with torch.no_grad():
                print("using mel pwg")
                Y_out_rn = Y_out_dn.cuda()
                Y_out_rn = torch.log10(torch.clamp(Y_out_rn, min=1e-10))
                Y_out_rn = (Y_out_rn - pwg_stats['mean']) / pwg_stats['scale']
                Z_in = torch.randn(1, 1, Y_out_rn.shape[2] * 320, device=Y_out_rn.device)
                Y_out_rn = F.pad(Y_out_rn,(2,2),mode='replicate')
                yhat = pwg_model(Z_in, Y_out_rn).view(-1)     

            yhat = yhat.cpu().numpy()
            

        ### output waveform
        sampling_rate = 16000
        MAX_WAV_VALUE = 32768.0
        directory_path = "/home/4TB_storage/hsinhao_storage/AVEL_align/cdvae-align-feature-sly-align/"
        if output_wav_type == "nl" :
            audio_path = directory_path + "0829_aligned_wav/NL01/" + output_file_name + ".wav"
        else :
            audio_path = directory_path + "0829_aligned_wav/EL01/" + "EL01_" + output_file_name.split("_")[--1] + ".wav"
        yhat = yhat * MAX_WAV_VALUE
        wavfile.write( audio_path, sampling_rate, yhat.astype('int16'))

        ### save aligned feature
        if output_wav_type == "nl" :
            filepath_pkl = directory_path + "0909_aligned_feat/NL01/"+ output_file_name + ".pkl"
        else :
            filepath_pkl = directory_path + "0909_aligned_feat/EL01/"+ "EL01_" + output_file_name.split("_")[-1] + ".pkl"
        for i in temp_feats :
            temp_feats[i] = temp_feats[i].cpu().detach().numpy()
        for i in temp_feats :
            print(type(temp_feats[i]))
        a_file = open(filepath_pkl, "wb")
        pickle.dump(temp_feats, a_file)
        a_file.close()

if __name__ == "__main__":
    import argparse
    import json
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/config_vae_vctk.json',
                    help='JSON file for configuration')
    args = parser.parse_args()
    # Load Config.
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    infer_config = config["infer_config"] 
    global data_config
    data_config = config["data_config"]    
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    feature_to_audio()







# one_frame_idx = np.empty([256]) 
# for i in range(256) :
#     one_frame_idx[i] = int(i)

# nl_time_idx = np.empty([0]) 
# for i in sly_align_nl :
#     print(i)
#     temp_nl_time_idx = one_frame_idx + 256*i
#     print(temp_nl_time_idx)
#     nl_time_idx = np.append( nl_time_idx , temp_nl_time_idx )

# print(nl_time_idx.shape)

# display_audio = np.empty([len(nl_time_idx)])
# for i in nl_time_idx :
#     temp_display_audio = y[i]
#     nl_time_idx = np.append( nl_time_idx , temp_nl_time_idx )


