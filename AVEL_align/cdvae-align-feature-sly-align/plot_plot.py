import numpy as np
import librosa
import matplotlib.pyplot as plt
from librosa.display import specshow

###
#############################################################################
sr = 16000
y, _ = librosa.load('../data/EL01/wav/EL01_001.wav', sr = sr)
### plot by librosa
# stft
S = librosa.stft(y, n_fft=1024, hop_length=256, win_length=1024)
S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)
# mel-scale
mS = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000,
                                    n_fft=1024, hop_length=256,
                                    win_length=1024)
#### MS = currently
mS_dB =librosa.power_to_db(np.abs(mS), ref=np.max)
print(mS_dB.shape)


sly_align_nl = np.load("./sly_aligned_NL_array/NL01_001.npy")
sly_align_el = np.load("./sly_aligned_EL_array/NL01_001.npy")
print(sly_align_el)
print('sly align lenth = ' , len(sly_align_el))

##### put index into it
sly_y = np.empty((128 , len(sly_align_el)))
for idx in range(len(sly_align_el)) :
    sly_y[:,idx] = mS_dB[:,sly_align_el[idx]]

sly_mS_dB = sly_y
print(sly_mS_dB.shape)


org_dtw_nl = np.load("./org_dtw_NL_array/NL01_001.npy")
org_dtw_el = np.load("./org_dtw_EL_array/NL01_001.npy")
print(org_dtw_el)
print('org dtw lenth = ' , len(org_dtw_el))

org_y = np.empty((128 , len(org_dtw_el)))
for idx in range(len(org_dtw_el)) :
    org_y[:,idx] = mS_dB[:,org_dtw_el[idx]]

org_mS_dB = org_y
print(sly_mS_dB.shape)

################
z, _ = librosa.load('../data/NL01/wav/NL01_001.wav', sr = sr)
# mel-scale
mS_z = librosa.feature.melspectrogram(y=z, sr=sr, n_mels=128, fmax=8000,
                                    n_fft=1024, hop_length=256,
                                    win_length=1024)
mS_z_dB =librosa.power_to_db(np.abs(mS_z), ref=np.max)

##### put index into it
sly_z = np.empty((128 , len(sly_align_nl)))
for idx in range(len(sly_align_nl)) :
    sly_z[:,idx] = mS_z_dB[:,sly_align_nl[idx]]

sly_nl_dB = sly_z
print(sly_nl_dB.shape)

##### put index into it
org_z = np.empty((128 , len(org_dtw_nl)))
for idx in range(len(org_dtw_nl)) :
    org_z[:,idx] = mS_z_dB[:,org_dtw_nl[idx]]

org_nl_dB = org_z
print(org_nl_dB.shape)



# create figure
fig, ax = plt.subplots(nrows=4, sharex=True, sharey=False, figsize=(12, 8))
fig.subplots_adjust(hspace=0.2, wspace=0.4)

# plot EL-mel-spec
specshow(mS_dB, sr=sr, ax = ax[0], hop_length=256, y_axis='mel', x_axis='s')
ax[0].set_title('EL-mel spec', fontsize = 14.0)
ax[0].label_outer()
# plot sly-mel-spec
specshow(sly_mS_dB, sr=sr, ax = ax[1], hop_length=256, y_axis='mel', x_axis='s')
ax[1].set_title('syl-aligned mel spec(U = 0,trim_silence = False)', fontsize = 14.0)
ax[1].label_outer()
# plot sly-mel-spec
specshow(org_mS_dB, sr=sr, ax = ax[2], hop_length=256, y_axis='mel', x_axis='s')
ax[2].set_title('org-dtw mel spec(U = 0,trim_silence = False)', fontsize = 14.0)
ax[2].label_outer()
# plot NL-mel-spec
specshow(mS_z_dB, sr=sr, ax = ax[3], hop_length=256, y_axis='mel', x_axis='s')
ax[3].set_title('NL-mel spec', fontsize = 14.0)
ax[3].label_outer()

fig.savefig('fig_spec_librosa.png', dpi=300)










# org_dtw_nl = np.load("./org_dtw_NL_array/NL01_001.npy")
# org_dtw_el = np.load("./org_dtw_EL_array/NL01_001.npy")
# print(org_dtw_el)

# ### plot by matplotlib
# # load waveform
# sr = 16000
# y, _ = librosa.load('../data/EL01/wav/EL01_001.wav', sr = sr)
# z, _ = librosa.load('../data/NL01/wav/NL01_001.wav', sr = sr)
# print("y")
# print(y.shape)
# print(len(y))

# x = list(range(1024)) # [0 1 2 ... 1024]
# ##### create an index list of sly_align
# new = []
# for content in sly_align_el :
#     temp = content * 256 + x
#     new = new + list(temp)
# # print("new")
# # print(len(new))
# # print(new[len(new)-1])

# ##### put index into it

# sly_y = np.empty(len(new))
# for idx in range(len(new)) :
#     sly_y[idx] = y[new[idx]]

# print(sly_y)
# print(len(sly_y))

# ##### create an index list of org_dtw
# new_org = []
# for content in org_dtw_el :
#     temp = content * 256 + x
#     new_org = new_org + list(temp)

# print("new_org")
# print(len(new_org))
# print(new_org[len(new_org)-1])
# ##### put index into it
# print(new_org)

# dtw_y = np.empty(len(new_org))
# for idx in range(len(new_org)) :
#     if (new_org[idx] < len(y)):
#         dtw_y[idx] = y[new_org[idx]]
#     else :
#         dtw_y[idx] = y[len(y)-1]
# print(dtw_y)


# recon_y , _ = librosa.load('./exp/result_wav/EL01_NL01_cdpjmel_el/EL01_NL01_001.wav', sr = sr)


# # setting figure 1111111111
# fig , ax = plt.subplots(nrows=4, ncols=1, sharex=True,
#                         sharey=True, figsize=(12, 8))
# fig.subplots_adjust(hspace=0.2, wspace=0.4)

# # plot image to sub figure
# ax[0].specgram(y, NFFT=1024, Fs=sr, noverlap = 1024-256)
# ax[0].set_title('EL', fontsize = 14.0)
# ax[0].label_outer()

# ax[1].specgram(sly_y, NFFT=1024, Fs=sr, noverlap = 1024-256)
# ax[1].set_title('EL with sly_align', fontsize = 14.0)
# ax[1].label_outer()

# ax[2].specgram(dtw_y, NFFT=1024, Fs=sr, noverlap=1024-256)
# ax[2].set_title('EL with org_dtw', fontsize = 14.0)
# ax[2].label_outer()

# ax[3].specgram(z, NFFT=1024, Fs=sr, noverlap=1024-256)
# ax[3].set_title('NL', fontsize = 14.0)
# ax[3].label_outer()

# plt.xlabel('Time (sec)',  fontsize = 14.0)
# plt.rcParams.update({'font.size': 14})
# # save figure
# fig.savefig('fig_1_spec_matplotlib.png', dpi=300)

# # setting figure 2222222222
# fig , ax = plt.subplots(nrows=2, ncols=1, sharex=True,
#                         sharey=True, figsize=(12, 8))
# fig.subplots_adjust(hspace=0.2, wspace=0.4)


# # plot image to sub figure
# ax[0].specgram(recon_y, NFFT=1024, Fs=sr, noverlap = 1024-256)
# ax[0].set_title('reconstructed from sly-align', fontsize = 14.0)
# ax[0].label_outer()

# ax[1].specgram(z, NFFT=1024, Fs=sr, noverlap = 1024-256)
# ax[1].set_title('NL', fontsize = 14.0)
# ax[1].label_outer()

# plt.xlabel('Time (sec)',  fontsize = 14.0)
# plt.rcParams.update({'font.size': 14})
# # save figure
# fig.savefig('fig_2_spec_matplotlib.png', dpi=300)
