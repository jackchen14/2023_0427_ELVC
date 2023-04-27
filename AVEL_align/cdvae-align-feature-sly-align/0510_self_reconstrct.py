import json

from pathlib import Path

from util.stft import STFT
from util.mel_spectrum import MelSpectrum
from util.wld_vocoder import Wld_vocoder
from WavLM_dir.WavLM import WavLM, WavLMConfig

import torch
import librosa
import ipdb

from librosa.effects import trim
from scipy.io import wavfile

def load_wav_to_torch(full_path, target_sampling_rate):
    """
    Loads wavdata into torch array
    """
    # data_sampling_rate, data = read(full_path)
    # data = data / MAX_WAV_VALUE
    data, data_sampling_rate = librosa.core.load(full_path, sr=target_sampling_rate,)

    if data_sampling_rate != target_sampling_rate:
        raise ValueError(
            "{} SR doesn't match target {} SR".format(
                data_sampling_rate, target_sampling_rate
            )
        )

    # if TRIM_SILENCE:
    #     data, _ = trim(data, top_db=25, frame_length=1024, hop_length=256,)
    return torch.from_numpy(data).float()


## input wav
sampling_rate = 16000
output_file_name = "NL01_001.wav"
wav_path = "/home/bioasp/Desktop/AVEL_align/data/NL01/wav/"+ output_file_name
audio = load_wav_to_torch(wav_path, sampling_rate)
print(audio.shape)
audio = audio.unsqueeze(0)
print(audio.shape)

wld_fn = Wld_vocoder(fft_size=400, 
                    shiftms=20, 
                    sampling_rate=sampling_rate, 
                    mcc_dim=24, mcc_alpha=0.455, 
                    minf0=40, maxf0=700,
                    cutoff_freq=70,
                    feat_stat=None).cuda()
## extract feature
feat = wld_fn(audio)
for key in feat :
    feat[key] = feat[key].float()
    print(key , feat[key].shape)

### synthesize
yhat = wld_fn.synthesis(feat, se_kind="mcc").view(-1)
yhat = yhat.cpu().numpy()

### output waveform
MAX_WAV_VALUE = 32768.0
audio_path = "/home/bioasp/Desktop/0510_self_reconstruct/" + output_file_name

yhat = yhat * MAX_WAV_VALUE
wavfile.write( audio_path, sampling_rate, yhat.astype('int16'))