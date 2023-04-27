import numpy as np
from scipy.signal import butter,filtfilt
from scipy.io import wavfile
from pathlib import Path


def butter_lowpass_filter(data, cutoff, fs, order):
    # Get the filter coefficients 
    b, a = butter(order, cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


order = 8
cutoff = 0.3
sampling_rate = 16000

str_wav_folder = "/home/bioasp/Desktop/AVEL_align/data/Zorg_NLEL_V2_dontouch/NL01/wav/"
wav_folder = Path(str_wav_folder)
for wav_path in sorted(list(wav_folder.glob('*'))):
    ## load wav
    fs , data  = wavfile.read(wav_path)

    ## low pass
    low_pass_data = butter_lowpass_filter(data, cutoff, fs, order)

    ## output low_pass audio
    str_wav_path = str(wav_path)
    output_file_name = str_wav_path.split('/')[-1]
    output_audio_path = "/home/bioasp/Desktop/AVEL_align/data/lowpass_NL01/wav/" + str(cutoff) + "/" + output_file_name
    wavfile.write( output_audio_path, sampling_rate, low_pass_data.astype('int16'))