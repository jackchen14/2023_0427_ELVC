from pathlib import Path
import os
import ipdb
org_data_root = "/home/bioasp/Desktop/ParallelWaveGAN-12_27/pwg_data/NL01/"
data_root = Path(org_data_root)

num = 1 
for speaker_name in sorted(list(data_root.glob('*.wav'))):
    print(speaker_name)
    str_num = str(num)
    str_new_file = "utt_" + str_num + ".wav"
    new_file_name = os.path.join( org_data_root , str_new_file)
    os.rename(speaker_name, new_file_name)
    num += 1
    # old_file = os.path.join("directory", data_num , test)
    # print(data_num)
    # print(test)

    

# ipdb.set_trace()
# old_file = os.path.join("directory", "a.txt")
# new_file = os.path.join("directory", "b.kml")
# os.rename(old_file, new_file)