## read audio
import librosa
audio = "EL01_001.wav"
full_path = "/home/bioasp/Downloads/aligned_wav-20220428T125541Z-001/aligned_wav/" + audio
data, data_sampling_rate = librosa.core.load(full_path, sr = 16000)
print(len(data))

full_path = "/home/bioasp/Desktop/aligned_wav/" + audio
data, data_sampling_rate = librosa.core.load(full_path, sr = 16000)
print(len(data))

