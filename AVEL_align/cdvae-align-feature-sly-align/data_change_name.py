import os
from pathlib import Path
# os.rename('a.txt', 'b.kml')
str_data_root = "/home/4TB_storage/hsinhao_storage/ParallelWaveGAN/pwg_aligned_nl_data/NL01"
data_root = Path(str_data_root)
count = 0
for speaker_name in sorted(list(data_root.glob('*_gen.wav'))):
    count += 1
    speaker_name_str = str(speaker_name)
    org_name = speaker_name_str
    new_name = speaker_name_str.split('_gen')[0] + '.wav'
    print("new name:\n" + new_name)
    os.rename(org_name,new_name)
print("count = " + str(count) )
    