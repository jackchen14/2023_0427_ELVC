import numpy as np
import librosa
import matplotlib.pyplot as plt
from librosa.display import specshow

sr = 16000
file_num = "244"
direct_name_list = ["EL01","SP_mel","wavlm_main","CFF_wavlm","NL01"]
for direct_name in direct_name_list :
    if direct_name != "EL01":
        print(direct_name)
        file_str = "02_18_thesis_audio/"+ direct_name + "/NL01_" + file_num + ".wav"
    else :
        print(direct_name)
        file_str = "02_18_thesis_audio/"+ direct_name + "/EL01_" + file_num + ".wav"
    y, sr = librosa.load(file_str)
    print("file lenth = ")
    print(str(len(y)/sr)+ " sec")


    # 繪製時域波形
    fig, ax = plt.subplots()
    duration = librosa.get_duration(y=y, sr=sr)
    t = np.linspace(0, duration, len(y))
    ax.plot(t, y)

    # 設置圖形屬性
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Amplitude')
    plt.savefig("02_18_thesis_audio/02_19_fig_plot/"+ direct_name + "_wav_" + file_num + ".png")
    # 繪製頻譜圖
    fig, ax = plt.subplots()
    NFFT = 1024
    noverlap = int(NFFT * 0.5)
    Pxx, freqs, bins, im = ax.specgram(y, Fs=sr, NFFT=NFFT, noverlap=noverlap)

    # 設置圖形屬性
    plt.ylim(0, 8000)
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Frequency (Hz)')
    plt.savefig("02_18_thesis_audio/02_19_fig_plot/"+ direct_name + "_spectro_" + file_num + ".png")

    plt.show()    

    import shutil
    print(file_str)
    shutil.copyfile(file_str, "02_18_thesis_audio/02_19_fig_plot/" + file_num + ".wav")


