import numpy as np
import matplotlib.pyplot as plt
import librosa

# 讀取音訊檔案
filename = "02_18_thesis_audio/NL01/NL01_241.wav"
y, sr = librosa.load(filename)
print("file lenth = ")
print(str(len(y)/sr)+ " sec")


# 繪製時域波形
fig, ax = plt.subplots()
duration = librosa.get_duration(y=y, sr=sr)
t = np.linspace(0, duration, len(y))
ax.plot(t, y)

# 設置圖形屬性
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude')
ax.set_title('Waveform of {}'.format(filename))

# 繪製頻譜圖
fig, ax = plt.subplots()
NFFT = 1024
noverlap = int(NFFT * 0.5)
Pxx, freqs, bins, im = ax.specgram(y, Fs=sr, NFFT=NFFT, noverlap=noverlap)

# 設置圖形屬性
ax.set_xlabel('Time [s]')
ax.set_ylabel('Frequency [Hz]')
ax.set_title('Spectrogram of {}'.format(filename))

plt.show()





