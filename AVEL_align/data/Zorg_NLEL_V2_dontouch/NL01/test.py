import torch
import librosa
data, data_sampling_rate = librosa.core.load("./wav/NL01_088.wav",sr = 16000)
print(data_sampling_rate)
print(len(data))
print("sec = " , len(data)/data_sampling_rate )