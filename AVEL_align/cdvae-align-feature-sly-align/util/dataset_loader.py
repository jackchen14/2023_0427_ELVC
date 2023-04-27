import os
import numpy as np
import torch
import random
from scipy.io.wavfile import read
from librosa.effects import  trim

from .stft import STFT
from .mel_spectrum import MelSpectrum
import ipdb

MAX_WAV_VALUE = 32768.0
MIN_SPEC_VALUE = 1e-10
TRIM_SILENCE = True

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip().split() for f in files]
    return files


def load_wav_to_torch(full_path, target_sampling_rate):
    """
    Loads wavdata into torch array
    """
    data_sampling_rate, data = read(full_path)

    if data_sampling_rate != target_sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR in {}".format(
            data_sampling_rate, target_sampling_rate, full_path))

    data = data / MAX_WAV_VALUE

    if TRIM_SILENCE:
        data,_ = trim(  data,
                        top_db=25,
                        frame_length=1024,
                        hop_length=256) 

    return torch.from_numpy(data).float()


class FeatDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, training_dir, segment_length, config):
        self.utt2path = dict(files_to_list(os.path.join(training_dir,'feats.scp')))
        self.utt2spk = files_to_list(os.path.join(training_dir,'utt2spk_id'))
        # print(training_dir)
        # ipdb.set_trace()
        # print("utt2path")
        # for utt in self.utt2path :
        #     print(utt)
        # print("utt2spk")
        # for utt,_ in self.utt2spk :
        #     print(utt)
        # print("_________________test__________________")
        # for utt,_ in self.utt2spk:
        #     if utt not in self.utt2path:
        #         print(utt)
        #         ipdb.set_trace()
        assert sum([(utt not in self.utt2path) for utt,_ in self.utt2spk]) == 0

        feat_kind = config.get('feature_kind', 'mel')
        pre_load  = config.get('pre_load', True)

        # random.seed(1234)
        # random.shuffle(self.utt2spk)

        feat_kinds = feat_kind.split('-')
        self.feat_kinds = []
        for feat_kind in feat_kinds:
            if feat_kind not in self.feat_kinds:
                self.feat_kinds.append(feat_kind)
            # self.feat_kinds.append(feat_kind)
        print(self.feat_kinds)
        

        if pre_load:
            utt2feat = []
            for utt,path in self.utt2path.items():
                _feat = torch.load(path, map_location='cpu')
                
                feat = [
                    _feat[key]
                    for key in self.feat_kinds
                    if key in _feat.keys()
                ]
                # print("matched feature key :")
                # count = 0
                # for key in self.feat_kinds :
                #     if key in _feat.keys() :
                #         print(key)
                #         count += 1
                # if count == 0 :
                #     print("all feature mismatched")
                # print("\n")
                # print("<utt> <feat>")
                # print(utt)
                # print(feat)
                ## breakpoint()
                

                utt2feat.append([utt,feat])
            self.utt2feat = dict(utt2feat)
            
            # print("nice")
        else:
            self.utt2feat = None
            print("bad qq")

        # ## test ##
        # utt, spk_id = self.utt2spk[1]
        # print("utt , spk_id")
        # print(utt)
        # print(spk_id)
        # print(len(self.utt2feat[utt]))
        # ## test ##
        self.segment_length = segment_length


    
    def __getitem__(self, index):
        # Read audio
        utt, spk_id = self.utt2spk[index]
       
        # print("------test------")  
        # print(index)   
        # print(utt) 
        if self.utt2feat is not None:
            feat = self.utt2feat[utt]
            # print("nisu")

        else:
            utt_path = self.utt2path[utt]
            _feat = torch.load(utt_path, map_location='cpu')
            # print("badu")
            feat = [
                    _feat[key]
                    for key in self.feat_kinds
                    if key in _feat.keys()
                ]

        # Take segment
        pos = [None] * len(feat)
        feat_start = None # keep feature alignment

        ## ex: total feature number = 5 , from 1st feature to 5th feature , find their length , named feat_length
        ## if each feat_length > segment length , crop it . else , pad it .
        for i in range(len(feat)):
            feat_length = feat[i].size(-1)
            if feat_length > self.segment_length:
                # Clip
                pos[i] = torch.arange(1,feat_length+1, dtype=torch.long)
                # print("pos i")
                # print(pos[i])
                pos[i][-1] *= -1
                # print("pos i ,-1")
                # print(pos[i][-1])
                # breakpoint()
                if feat_start is None:
                    max_audio_start = feat_length - self.segment_length
                    feat_start = random.randint(0, max_audio_start)
                    feat_end = feat_start + self.segment_length
                feat[i] = feat[i][...,feat_start:feat_end]
                pos[i] = pos[i][feat_start:feat_end]

            else:
                # Padding
                pos[i] = torch.arange(1,feat_length+1, dtype=torch.long)
                pos[i][-1] *= -1
                padding = self.segment_length - feat_length
                feat[i] = torch.nn.functional.pad(feat[i], (0, padding), 'constant').data
                pos[i] = torch.nn.functional.pad(pos[i], (0, padding), 'constant').data
                # ipdb.set_trace()
        spk_id_out = torch.ones(1, dtype=torch.long) * int(spk_id)
        feat.append(spk_id_out)
        # print(len(feat))
        # print("len pos")
        # print(len(pos))
        if 'pos' in self.feat_kinds:
            feat = feat + pos

        # print(len(feat))
        # print("__getitem__ return feat , feat = feat + position ")
        # breakpoint()
        return feat

    def __len__(self):
        return len(self.utt2spk)


class WavDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, training_dir, segment_length, config):
        self.utt2path = dict(files_to_list(os.path.join(training_dir,'wav.scp')))
        self.utt2spk = files_to_list(os.path.join(training_dir,'utt2spk_id'))
        
        stat_dict       = config.get('statistic_file', None)
        feat_kind       = config.get('feature_kind', 'mel')
        filter_length   = config.get('filter_length', 1024)
        hop_length      = config.get('hop_length', 256)
        win_length      = config.get('win_length', 1024)
        n_mel_channels  = config.get('n_mel_channels', 80)
        sampling_rate   = config.get('sampling_rate', 24000)
        mel_fmin        = config.get('mel_fmin', 80)
        mel_fmax        = config.get('mel_fmax', 7600)
        device          = config.get('device', 'cpu')

        random.seed(1234)
        random.shuffle(self.utt2spk)

        if stat_dict is not None and stat_dict != '':
            print('Use Stat. \"{}\"'.format(stat_dict))
            feat_stat = torch.load(stat_dict)
        else:
            feat_stat = None

        self.device = torch.device(device)
        if 'stft' in feat_kind:
            self.feat_fn = STFT( filter_length=filter_length,
                                 hop_length=hop_length, 
                                 win_length=win_length, 
                                 window='hann',
                                 feat_stat=feat_stat).to(self.device)

        elif 'mel' in feat_kind:
            self.feat_fn = MelSpectrum( filter_length=filter_length,
                                        hop_length=hop_length,
                                        win_length=win_length,
                                        n_mel_channels=n_mel_channels,
                                        sampling_rate=sampling_rate,
                                        mel_fmin=mel_fmin, mel_fmax=mel_fmax,
                                        feat_stat=feat_stat).to(self.device)

        self.segment_length = segment_length
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.feat_kind = feat_kind
        

    def get_feat(self, audio):
        audio = audio.unsqueeze(0)
        with torch.no_grad():
            feat = self.feat_fn(audio)
            feat = self.feat_fn.normalize(feat)
        return feat.squeeze(0)

    def __getitem__(self, index):
        # Read audio
        utt, spk_id = self.utt2spk[index]
        utt_path = self.utt2path[utt]

        audio = load_wav_to_torch(utt_path, self.sampling_rate).float().to(self.device)

        # Take segment
        if audio.size(0) > self.segment_length:
            pos = torch.arange(1,audio.size(0)//self.hop_length+2, dtype=torch.long, device=self.device)
            pos[-1] *= -1
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment_length]
            pos = pos[audio_start//self.hop_length:(audio_start+self.segment_length)//self.hop_length]
        else:
            pos = torch.arange(1,audio.size(0)//self.hop_length+2, dtype=torch.long, device=self.device)
            pos[-1] *= -1
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data


        if self.feat_kind == 'wav':
            audio = audio.unsqueeze(0)
            spk_id_out = torch.ones(audio.size(-1), dtype=torch.long, device=self.device) * int(spk_id)
            return audio, spk_id_out

        feat = self.get_feat(audio)[:,:self.segment_length // self.hop_length]
        spk_id_out = torch.ones(1, dtype=torch.long, device=self.device) * int(spk_id)

        if 'wav' in self.feat_kind:
            audio = audio.unsqueeze(0)
            return [feat, audio], spk_id_out

        if 'pos' in self.feat_kind:
            pos = torch.nn.functional.pad(pos, (0, feat.size(-1) - pos.size(0)), 'constant').data
            feat = feat * pos.ne(0).float().unsqueeze(0)
            return feat, spk_id_out, pos

        return feat, spk_id_out

    def __len__(self):
        return len(self.utt2spk)

