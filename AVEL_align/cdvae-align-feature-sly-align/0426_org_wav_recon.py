import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
from util.wld_vocoder import Wld_vocoder
from librosa.display import specshow
from pathlib import Path
from scipy.io import wavfile
import os
import torch.nn.functional as F

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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


# load feature
output_wav_type = "nl"
output_wav_type_big = "NL"

# output_wav_type = "el"
# output_wav_type_big = "EL"


for data_num in range(30) :
    output_file_name = "NL01_" + "%03d" % (data_num+1)
    if output_wav_type == "nl" :
        feat_path = "/home/4TB_storage/hsinhao_storage/AVEL_align/cdvae-align-feature-sly-align/features/train_nl/" + output_file_name + ".pt"
    else :
        feat_path = "/home/4TB_storage/hsinhao_storage/AVEL_align/cdvae-align-feature-sly-align/features/tmhint_" + output_wav_type + "_train/" + "EL01_" + output_file_name.split("_")[-1] + ".pt"
    feats = torch.load(feat_path)
    for feat_name in feats :
        print(feat_name , feats[feat_name].shape)

    ## insert aligned index into feature 
    for i in feats :
        if i == 'WavLM' :
            abcde = 0
        elif i == 'f0' :
            feats[i] = feats[i].reshape(1,feats[i].shape[0])
        else :
            feats[i] = feats[i].reshape(1,feats[i].shape[0],feats[i].shape[1])

    temp_feats = feats
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

        ### mel vocoder ###
        with torch.no_grad():
            print("using mel pwg")
            Y_out_rn = temp_feats["mel"].cuda()
            Y_out_rn = torch.log10(torch.clamp(Y_out_rn, min=1e-10))
            Y_out_rn = (Y_out_rn - pwg_stats['mean']) / pwg_stats['scale']
            Z_in = torch.randn(1, 1, Y_out_rn.shape[2] * 320, device=Y_out_rn.device)
            Y_out_rn = F.pad(Y_out_rn,(2,2),mode='replicate')
            yhat = pwg_model(Z_in, Y_out_rn).view(-1)     

        yhat = yhat.cpu().numpy()
        

    ### output waveform
    sampling_rate = 16000
    MAX_WAV_VALUE = 32768.0
    if output_wav_type == "nl" :
        audio_path = "/home/4TB_storage/hsinhao_storage/AVEL_align/cdvae-align-feature-sly-align/0829_org_wav/" + output_file_name + ".wav"
    else :
        audio_path = "/home/4TB_storage/hsinhao_storage/AVEL_align/cdvae-align-feature-sly-align/0829_org_wav/" + "EL01_" + output_file_name.split("_")[-1] + ".wav"
    yhat = yhat * MAX_WAV_VALUE

    wavfile.write( audio_path, sampling_rate, yhat.astype('int16'))








