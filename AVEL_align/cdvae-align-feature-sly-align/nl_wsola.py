import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from sprocket.speech import FeatureExtractor, Synthesizer
from sprocket.util import HDF5
import ipdb
import math
# from sprocket.model import GV, F0statistics,Statistics
from sprocket.speech import WSOLA
# from misc import low_cut_filter, low_pass_filter, convert_to_continuos_f0
# from yml import SpeakerYML
dirname = 'finetune_wNL01'

training_dir = "/home/4TB_storage/hsinhao_storage/AVEL_align/data/Zorg_NLEL_V2_dontouch/NL01/wav"
training_dir = Path(training_dir)
output_dir = "/home/4TB_storage/hsinhao_storage/AVEL_align/data/finetune_wNL01/wav"


for data_file in sorted(list(training_dir.glob('*.wav'))):
    print("---------------")
    print(data_file)
    # f = line.rstrip()
    # el_f = f.replace('NL','EL')
    # print(f,el_f)
    data_file = str(data_file)
    nl_wavf = data_file
    el_wavf = data_file.replace('NL01/wav/NL01','EL01/wav/EL01')
    print(el_wavf)

    fs_n,x_n = wavfile.read(nl_wavf)
    fs_e,x_e = wavfile.read(el_wavf)
    s_rate = len(x_n)/len(x_e)
    # ipdb.set_trace()
    # s_rate = math.ceil(s_rate * 100) / 100.0
    print("length & rate")
    print(len(x_n))
    print(len(x_e))
    print(s_rate)
    wsola = WSOLA(fs_n,s_rate)
    wsolaed_x = wsola.duration_modification(x_n)
    assert int(len(x_n) / s_rate) == len(wsolaed_x)
    # ipdb.set_trace()
    if len(wsolaed_x) >= len(x_e):
        wsolaed_x = wsolaed_x[:len(x_e)]
    else:
        pad_wav = np.zeros([len(x_e)])
        pad_wav[:len(wsolaed_x)] = wsolaed_x
        wsolaed_x = pad_wav 
    assert len(wsolaed_x)==len(x_e)
    # ipdb.set_trace()
    wavpath = os.path.join(output_dir ,data_file.split('/')[-1])
    wavfile.write(wavpath, fs_n, np.array(wsolaed_x, dtype=np.int16))
    print(wavpath)
