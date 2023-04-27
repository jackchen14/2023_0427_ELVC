# 2021-10-18 ymchiqq plot_spec

import numpy as np
import librosa
import matplotlib.pyplot as plt
from librosa.display import specshow

### plot by matplotlib
# load waveform
sr = 16000
y, _ = librosa.load('./wav/NL01_001.wav', sr=sr)
print("y")
print(y.shape)
print(y)
# setting figure
fig , ax = plt.subplots(nrows=2, ncols=1, sharex=True,
                        sharey=True, figsize=(12, 8))
fig.subplots_adjust(hspace=0.2, wspace=0.4)

# plot image to sub figure
ax[0].specgram(y, NFFT=1024, Fs=sr, noverlap=1024-256)
ax[0].set_title('spec 1', fontsize = 14.0)
ax[0].label_outer()

ax[1].specgram(y, NFFT=1024, Fs=sr, noverlap=1024-256)
ax[1].set_title('spec 2', fontsize = 14.0)
ax[1].label_outer()
plt.xlabel('Time (sec)',  fontsize = 14.0)
plt.rcParams.update({'font.size': 14})
# save figure
fig.savefig('fig_spec_matplotlib.png', dpi=300)

# ### plot by librosa
# # stft
# S = librosa.stft(y, n_fft=1024, hop_length=256, win_length=1024)
# S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)
# # mel-scale
# mS = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000,
#                                     n_fft=1024, hop_length=256,
#                                     win_length=1024)
# mS_dB =librosa.power_to_db(np.abs(mS), ref=np.max)

# # create figure
# fig, ax = plt.subplots(nrows=2, sharex=True, sharey=False, figsize=(12, 8))
# fig.subplots_adjust(hspace=0.2, wspace=0.4)

# # plot stft
# specshow(S_dB, sr=sr, ax=ax[0], hop_length=256, y_axis='mel', x_axis='s')
# ax[0].set_title('spec stft', fontsize = 14.0)
# ax[0].label_outer()
# # plot mel-spec
# specshow(mS_dB, sr=sr, ax=ax[1], hop_length=256, y_axis='mel', x_axis='s')
# ax[1].set_title('spec mel', fontsize = 14.0)
# ax[1].label_outer()

# fig.savefig('fig_spec_librosa.png', dpi=300)