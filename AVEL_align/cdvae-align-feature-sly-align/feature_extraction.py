import json

from pathlib import Path

from util.stft import STFT
from util.mel_spectrum import MelSpectrum
from util.wld_vocoder import Wld_vocoder
from WavLM_dir.WavLM import WavLM, WavLMConfig

import torch
import librosa
import ipdb
import copy

from librosa.effects import trim
import torch.nn.functional as F


MAX_WAV_VALUE = 32768.0
MAX_MAT_NUM = 30000
MIN_SPEC_VALUE = 1e-10
#TRIM_SILENCE = True
TRIM_SILENCE = False

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(str(filename), "r") as f:
        files = f.readlines()

    files = [f.rstrip().split() for f in files]
    return files


def load_wav_to_torch(full_path, target_sampling_rate):
    """
    Loads wavdata into torch array
    """
    # data_sampling_rate, data = read(full_path)
    # data = data / MAX_WAV_VALUE
    data, data_sampling_rate = librosa.core.load(full_path, sr=target_sampling_rate,)

    if data_sampling_rate != target_sampling_rate:
        raise ValueError(
            "{} SR doesn't match target {} SR".format(
                data_sampling_rate, target_sampling_rate
            )
        )

    if TRIM_SILENCE:
        data, _ = trim(data, top_db=25, frame_length=1024, hop_length=256,)
    return torch.from_numpy(data).float()

checkpoint = torch.load('/home/4TB_storage/hsinhao_storage/AVEL_align/cdvae-align-feature-sly-align/WavLM-Base.pt')
cfg = WavLMConfig(checkpoint['cfg'])
wavlm_model = WavLM(cfg)
wavlm_model.load_state_dict(checkpoint['model'])
wavlm_model.eval()
# def WavLM_extraction(wav_input_16khz):
#     # extract the representation of last layer
#     rep = model.extract_features(wav_input_16khz)[0]
#     rep = rep.detach()
#     return rep


def main(config):
    training_dir = config.get("training_dir", "")
    feature_kind = config.get("feature_kind", "mel")
    feature_dir = config.get("feature_dir", "none")
    fft_size = config.get("fft_size", 1024)
    hop_length = config.get("hop_length", 256)
    shiftms = config.get("shiftms", 5)
    win_length = config.get("win_length", 1024)
    n_mel_channels = config.get("n_mel_channels", 80)
    mcc_dim = config.get("mcc_dim", 24)
    mcc_alpha = config.get("mcc_alpha", 0.455)
    sampling_rate = config.get("sampling_rate", 24000)
    mel_fmin = config.get("mel_fmin", 80)
    mel_fmax = config.get("mel_fmax", 7600)
    f0_fmin = config.get("f0_fmin", 80)
    f0_fmax = config.get("f0_fmax", 7600)
    WavLM_flag = config.get("WavLM_flag", "No")
    # f0_fmin = config.get("f0_fmin", 40)
    # f0_fmax = config.get("f0_fmax", 700)
    cutoff_freq = config.get("cutoff_freq", 70)

    training_dir = Path(training_dir)
    assert feature_dir is not None and feature_dir not in ["none"]
    feature_dir = Path(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    feature_kinds = feature_kind.split("-")
    print("Extract: {}".format(feature_kinds))
    func_kinds = []

    if "stft" in feature_kinds:
        stft_fn = STFT(
            filter_length=fft_size,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
        )
        func_kinds.append(["stft", stft_fn])

    # if "mel" in feature_kinds:
    #     mel_fn = MelSpectrum(
    #         filter_length=fft_size,
    #         hop_length=hop_length,
    #         win_length=win_length,
    #         n_mel_channels=n_mel_channels,
    #         sampling_rate=sampling_rate,
    #         mel_fmin=mel_fmin,
    #         mel_fmax=mel_fmax,
    #     )
    #     func_kinds.append(["mel", mel_fn])

    if "mel" in feature_kinds:
        mel_fn = MelSpectrum(
            filter_length=fft_size,# 400
            hop_length=hop_length,# 320
            win_length=win_length,# 400
            n_mel_channels=n_mel_channels,
            sampling_rate=sampling_rate,
            mel_fmin=mel_fmin,
            mel_fmax=mel_fmax,
        )
        func_kinds.append(["mel", mel_fn])

    list_ftypes = ["sp", "mcc", "f0", "ap", "wld"]
    if sum([(f in feature_kinds) for f in list_ftypes]) > 0:
        wld_fn = Wld_vocoder(
            fft_size=fft_size,
            shiftms=shiftms,
            sampling_rate=sampling_rate,
            mcc_dim=mcc_dim,
            mcc_alpha=mcc_alpha,
            minf0=f0_fmin,
            maxf0=f0_fmax,
            cutoff_freq=cutoff_freq,
        )
        func_kinds.append(["wld", wld_fn])

    data_list = files_to_list(training_dir / "wav.scp")
    feat_scp = open(training_dir / "feats.scp", "w")

    for data_name, data_path in data_list:
        print("Data : {}".format(data_name), end=" " * 30 + "\r")

        audio = load_wav_to_torch(data_path, sampling_rate)
        audio = audio.unsqueeze(0)

        feat = dict()
        for func_kind, feat_fn in func_kinds:
            _feat = feat_fn(audio)

            for key, val in _feat.items():
                if key in feature_kinds:
                    feat[key] = val.squeeze(0).float()
                    # print(key)
                    # print(feat[key].shape)

        if WavLM_flag == "WavLM_Yes" :
            # wavlm_model = WavLM(cfg)
            # wavlm_model.load_state_dict(checkpoint['model'])
            # wavlm_model.eval()
            with torch.no_grad():
                rep = wavlm_model.extract_features(audio)[0]
                rep = rep.detach()
                WavLM_feat = rep.transpose(1,2)
                # WavLM_feat = F.pad(WavLM_feat,(1,1),mode='replicate')
                WavLM_feat = WavLM_feat.squeeze(0).float()
                # print(WavLM_feat)              
                # print(WavLM_feat.shape)
                # print(WavLM_feat)
                # start = copy.deepcopy(WavLM_feat[:,0])
                # start = torch.reshape(start,(768,1))
                # stop = copy.deepcopy(WavLM_feat[:,-1])
                # stop = torch.reshape(stop,(768,1))
                # print(start)
                # print(WavLM_feat.shape)
                # print(type(WavLM_feat))
                # WavLM_feat = torch.cat((start,WavLM_feat),1)
                # WavLM_feat = torch.cat((WavLM_feat,stop),1)
                # ipdb.set_trace()
                feat["WavLM"] = WavLM_feat
                del rep
                del WavLM_feat

        # ipdb.set_trace()
        
        output_path = feature_dir / "{}.pt".format(data_name)
        feat_scp.write("{} {}\n".format(data_name, str(output_path.absolute())))
        # print(output_path)
        # ipdb.set_trace()
        # print("audio", audio.shape)
        # for i in feat :
        #     print(i , feat[i].shape)
        # print("mel")
        # print(feat["mel"])
        # ipdb.set_trace()
        torch.save(feat, output_path)
        del feat

    feat_scp.close()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="conf/config_feature.json",
        help="JSON file for configuration",
    )
    parser.add_argument(
        "-T", "--training_dir", type=str, default=None, help="Traininig dictionary path"
    )
    parser.add_argument(
        "-F", "--feature_dir", type=str, default=None, help="Feature data dir"
    )
    parser.add_argument(
        "-K", "--feature_kind", type=str, default=None, help="Feature kind"
    )
    parser.add_argument(
        "-W", "--WavLM_flag", type=str, default=None, help="WavLM_flag"
    )
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    data_config = config["data_config"]

    if args.training_dir is not None:
        data_config["training_dir"] = args.training_dir
    if args.feature_dir is not None:
        data_config["feature_dir"] = args.feature_dir
    if args.feature_kind is not None:
        data_config["feature_kind"] = args.feature_kind
    data_config["WavLM_flag"] = args.WavLM_flag
    main(data_config)
