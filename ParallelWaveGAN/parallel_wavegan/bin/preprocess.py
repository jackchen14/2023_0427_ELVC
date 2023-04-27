#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Perform preprocessing and raw feature extraction."""

import argparse
import logging
import os

import librosa
import numpy as np
import soundfile as sf
import yaml
import torch

from tqdm import tqdm

from parallel_wavegan.datasets import AudioDataset
from parallel_wavegan.datasets import AudioSCPDataset
from parallel_wavegan.utils import write_hdf5
# from sprocket.speech import FeatureExtractor
from WavLM_dir.WavLM import WavLM, WavLMConfig
import pickle


def logmelfilterbank(
    audio,
    sampling_rate,
    fft_size=1024,
    hop_size=256,
    win_length=None,
    window="hann",
    num_mels=80,
    fmin=None,
    fmax=None,
    eps=1e-10,
    log_base=10.0,
):
    """Compute log-Mel filterbank feature.

#     Args:
#         audio (ndarray): Audio signal (T,).
#         sampling_rate (int): Sampling rate.
#         fft_size (int): FFT size.
#         hop_size (int): Hop size.
#         win_length (int): Window length. If set to None, it will be the same as fft_size.
#         window (str): Window function type.
#         num_mels (int): Number of mel basis.
#         fmin (int): Minimum frequency in mel basis calculation.
#         fmax (int): Maximum frequency in mel basis calculation.
#         eps (float): Epsilon value to avoid inf in log calculation.
#         log_base (float): Log base. If set to None, use np.log.

#     Returns:
#         ndarray: Log Mel filterbank feature (#frames, num_mels).

#     """
    # get amplitude spectrogram
    x_stft = librosa.stft(
        audio,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        pad_mode="reflect",
    )
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)
    mel = np.maximum(eps, np.dot(spc, mel_basis.T))

    if log_base is None:
        return np.log(mel)
    elif log_base == 10.0:
        return np.log10(mel)
    elif log_base == 2.0:
        return np.log2(mel)
    else:
        raise ValueError(f"{log_base} is not supported.")

def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features (See detail in parallel_wavegan/bin/preprocess.py)."
    )
    parser.add_argument(
        "--wav-scp",
        "--scp",
        default=None,
        type=str,
        help="kaldi-style wav.scp file. you need to specify either scp or rootdir.",
    )
    parser.add_argument(
        "--segments",
        default=None,
        type=str,
        help="kaldi-style segments file. if use, you must to specify both scp and segments.",
    )
    parser.add_argument(
        "--rootdir",
        default=None,
        type=str,
        help="directory including wav files. you need to specify either scp or rootdir.",
    )
    parser.add_argument(
        "--dumpdir",
        type=str,
        required=True,
        help="directory to dump feature files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # check arguments
    if (args.wav_scp is not None and args.rootdir is not None) or (
        args.wav_scp is None and args.rootdir is None
    ):
        raise ValueError("Please specify either --rootdir or --wav-scp.")

    # get dataset
    if args.rootdir is not None:
        dataset = AudioDataset(
            args.rootdir,
            "*.wav",
            audio_load_fn=sf.read,
            return_utt_id=True,
        )
        print("a")
    else:
        dataset = AudioSCPDataset(
            args.wav_scp,
            segments=args.segments,
            return_utt_id=True,
            return_sampling_rate=True,
        )
        print("b")

    # check directly existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir, exist_ok=True)

    # feature_extractor = FeatureExtractor(
    #         analyzer="world",
    #         fs=config['sampling_rate'],
    #         shiftms=config['shiftms'],
    #         minf0=40,
    #         maxf0=700,
    #         fftl=config['fft_size'])
    checkpoint = torch.load('/home/4TB_storage/hsinhao_storage/AVEL_align/cdvae-align-feature-sly-align/WavLM-Base.pt')
    cfg = WavLMConfig(checkpoint['cfg'])
    wavlm_model = WavLM(cfg)
    wavlm_model.load_state_dict(checkpoint['model'])
    wavlm_model.eval()
    count_a = 0
    for utt_id, (audio, fs) in tqdm(dataset):
        # check
        print(utt_id)
        print(audio)
        assert len(audio.shape) == 1, f"{utt_id} seems to be multi-channel signal."
        assert (
            np.abs(audio).max() <= 1.0
        ), f"{utt_id} seems to be different from 16 bit PCM."
        assert (
            fs == config["sampling_rate"]
        ), f"{utt_id} seems to have a different sampling rate."

        # trim silence
        if config["trim_silence"]:
            audio, _ = librosa.effects.trim(
                audio,
                top_db=config["trim_threshold_in_db"],
                frame_length=config["trim_frame_size"],
                hop_length=config["trim_hop_size"],
            )

        if "sampling_rate_for_feats" not in config:
            x = audio
            sampling_rate = config["sampling_rate"]
            hop_size = config["hop_size"]
            print("yes")
        else:
            # NOTE(kan-bayashi): this procedure enables to train the model with different
            #   sampling rate for feature and audio, e.g., training with mel extracted
            #   using 16 kHz audio and 24 kHz audio as a target waveform
            x = librosa.resample(audio, fs, config["sampling_rate_for_feats"])
            sampling_rate = config["sampling_rate_for_feats"]
            assert (
                config["hop_size"] * config["sampling_rate_for_feats"] % fs == 0
            ), "hop_size must be int value. please check sampling_rate_for_feats is correct."
            hop_size = config["hop_size"] * config["sampling_rate_for_feats"] // fs
            print("sad")

        # extract feature
        ############# concate < WavLM + mel > extraction ###############
        t = torch.from_numpy(x)
        t = t.unsqueeze(0)
        finetune_PWG = False
        converted_feat_finetune_PWG = False
        converted_concated_finetune_PWG = True
        if (converted_concated_finetune_PWG == True):
            print("converted_concated_feat_PWG")
            dir_path = "/home/4TB_storage/hsinhao_storage/AVEL_align/cdvae-align-feature-sly-align/converted_concate_WavMel/"
            data_path = "EL01_" + str(utt_id.split("_")[-1]) +".npy"
            filepath_np = os.path.join(dir_path, data_path)
            WavMel = np.load(filepath_np)
            WavMel = np.squeeze(WavMel, axis=0)
            WavMel = np.transpose(WavMel)
            print("WavMel")
            print(WavMel.shape) 
            concate_WavMel = WavMel

        elif (finetune_PWG == True) and (converted_feat_finetune_PWG == False) :
            print("finetune_PWG")
            dir_path = "/home/4TB_storage/hsinhao_storage/AVEL_align/cdvae-align-feature-sly-align/0909_aligned_feat/EL01/"
            data_path = "EL01_" + str(utt_id.split("_")[-1]) +".pkl"
            filepath_pkl = os.path.join(dir_path, data_path)
            a_file = open(dir_path + data_path, "rb")
            finetune_feat = pickle.load(a_file)
            # print(finetune_feat.shape)
            WavLM_feat = np.squeeze(finetune_feat["WavLM"], axis=0)
            WavLM_feat = np.transpose(WavLM_feat)
            print("WavLM_feat")
            print(WavLM_feat.shape)          
            a_file.close()
        elif converted_feat_finetune_PWG == True :
            print("converted_feat_finetune_PWG")
            dir_path = "/home/4TB_storage/hsinhao_storage/AVEL_align/cdvae-align-feature-sly-align/converted_WavLM/"
            data_path = "EL01_" + str(utt_id.split("_")[-1]) +".npy"
            filepath_np = os.path.join(dir_path, data_path)
            WavLM_feat = np.load(filepath_np)
            WavLM_feat = np.squeeze(WavLM_feat, axis=0)
            WavLM_feat = np.transpose(WavLM_feat)
            print("WavLM_feat")
            print(WavLM_feat.shape) 

        else :
            with torch.no_grad():   
                rep = wavlm_model.extract_features(t)[0]
                rep = rep.detach()
                WavLM_feat = rep.transpose(1,2)
                # WavLM_feat = F.pad(WavLM_feat,(1,1),mode='replicate')
                WavLM_feat = WavLM_feat.squeeze(0).float()
                WavLM_feat = WavLM_feat.transpose(0,1)
                WavLM_feat = WavLM_feat.numpy()
            # print("WavLM_feat")
            # print(type(WavLM_feat))
            # print(WavLM_feat.shape)
            
        # mel = logmelfilterbank(
        #     x,
        #     sampling_rate=sampling_rate,
        #     hop_size=hop_size,
        #     fft_size=config["fft_size"],
        #     win_length=config["win_length"],
        #     window=config["window"],
        #     num_mels=config["num_mels"],
        #     fmin=config["fmin"], ## 40
        #     fmax=config["fmax"], ## 700
        #     # keep compatibility
        #     log_base=config.get("log_base", 10.0),
        # )
        # # make sure the audio length and feature length are matched
        # print(type(WavLM_feat))
        # print(WavLM_feat.shape)
        # print(type(mel))
        # print(mel.shape)
        # if WavLM_feat.shape[0] == (mel.shape[0] - 2):
        #     mel = mel[1:-1]
        # else:
        #     mel = mel[0:-1]
        # print(mel.shape)
        # concate_WavMel = np.concatenate((WavLM_feat,mel),axis = 1)
        # print(concate_WavMel.shape)
        
        audio = np.pad(audio, (0, config["fft_size"]), mode="reflect")
        audio = audio[: len(concate_WavMel) * config["hop_size"]]
        print(len(audio))
        print(len(concate_WavMel)* config["hop_size"])
        assert len(concate_WavMel) * config["hop_size"] == len(audio)
        # import ipdb
        # ipdb.set_trace()

        # apply global gain
        if config["global_gain_scale"] > 0.0:
            audio *= config["global_gain_scale"]
        if np.abs(audio).max() >= 1.0:
            logging.warn(
                f"{utt_id} causes clipping. "
                f"it is better to re-consider global gain scale."
            )
            continue
        # save
        if config["format"] == "hdf5":
            write_hdf5(
                os.path.join(args.dumpdir, f"{utt_id}.h5"),
                "wave",
                audio.astype(np.float32),
            )
            write_hdf5(
                os.path.join(args.dumpdir, f"{utt_id}.h5"),
                "feats",
                concate_WavMel.astype(np.float32),
            )
        elif config["format"] == "npy":
            np.save(
                os.path.join(args.dumpdir, f"{utt_id}-wave.npy"),
                audio.astype(np.float32),
                allow_pickle=False,
            )
            np.save(
                os.path.join(args.dumpdir, f"{utt_id}-feats.npy"),
                concate_WavMel.astype(np.float32),
                allow_pickle=False,
            )
        else:
            raise ValueError("support only hdf5 or npy format.")
    
        ############# WavLM extraction ###############
        
        # t = torch.from_numpy(x)
        # t = t.unsqueeze(0)
        # finetune_PWG = False
        # converted_feat_finetune_PWG = True
        # if (finetune_PWG == True) and (converted_feat_finetune_PWG == False) :
        #     print("finetune_PWG")
        #     dir_path = "/home/4TB_storage/hsinhao_storage/AVEL_align/cdvae-align-feature-sly-align/0909_aligned_feat/EL01/"
        #     data_path = "EL01_" + str(utt_id.split("_")[-1]) +".pkl"
        #     filepath_pkl = os.path.join(dir_path, data_path)
        #     a_file = open(dir_path + data_path, "rb")
        #     finetune_feat = pickle.load(a_file)
        #     # print(finetune_feat.shape)
        #     WavLM_feat = np.squeeze(finetune_feat["WavLM"], axis=0)
        #     WavLM_feat = np.transpose(WavLM_feat)
        #     print("WavLM_feat")
        #     print(WavLM_feat.shape)          
        #     a_file.close()
        # elif converted_feat_finetune_PWG == True :
        #     print("converted_feat_finetune_PWG")
        #     dir_path = "/home/4TB_storage/hsinhao_storage/AVEL_align/cdvae-align-feature-sly-align/converted_WavLM/"
        #     data_path = "EL01_" + str(utt_id.split("_")[-1]) +".npy"
        #     filepath_np = os.path.join(dir_path, data_path)
        #     WavLM_feat = np.load(filepath_np)
        #     WavLM_feat = np.squeeze(WavLM_feat, axis=0)
        #     WavLM_feat = np.transpose(WavLM_feat)
        #     print("WavLM_feat")
        #     print(WavLM_feat.shape) 

        # else :
        #     with torch.no_grad():   
        #         rep = wavlm_model.extract_features(t)[0]
        #         rep = rep.detach()
        #         WavLM_feat = rep.transpose(1,2)
        #         # WavLM_feat = F.pad(WavLM_feat,(1,1),mode='replicate')
        #         WavLM_feat = WavLM_feat.squeeze(0).float()
        #         WavLM_feat = WavLM_feat.transpose(0,1)
        #         WavLM_feat = WavLM_feat.numpy()
        #     print("WavLM_feat")
        #     print(type(WavLM_feat))
        #     print(WavLM_feat.shape)
            

        # # make sure the audio length and feature length are matched
        # count_a += 1
        # print(count_a)
        # audio = np.pad(audio, (0, config["fft_size"]), mode="reflect")
        # audio = audio[: len(WavLM_feat) * config["hop_size"]]
        # print(len(audio))
        # print(len(WavLM_feat)* config["hop_size"])
        # assert len(WavLM_feat) * config["hop_size"] == len(audio)
        # # import ipdb
        # # ipdb.set_trace()

        # # apply global gain
        # if config["global_gain_scale"] > 0.0:
        #     audio *= config["global_gain_scale"]
        # if np.abs(audio).max() >= 1.0:
        #     logging.warn(
        #         f"{utt_id} causes clipping. "
        #         f"it is better to re-consider global gain scale."
        #     )
        #     continue
        # # save
        # if config["format"] == "hdf5":
        #     write_hdf5(
        #         os.path.join(args.dumpdir, f"{utt_id}.h5"),
        #         "wave",
        #         audio.astype(np.float32),
        #     )
        #     write_hdf5(
        #         os.path.join(args.dumpdir, f"{utt_id}.h5"),
        #         "feats",
        #         WavLM_feat.astype(np.float32),
        #     )
        # elif config["format"] == "npy":
        #     np.save(
        #         os.path.join(args.dumpdir, f"{utt_id}-wave.npy"),
        #         audio.astype(np.float32),
        #         allow_pickle=False,
        #     )
        #     np.save(
        #         os.path.join(args.dumpdir, f"{utt_id}-feats.npy"),
        #         WavLM_feat.astype(np.float32),
        #         allow_pickle=False,
        #     )
        # else:
        #     raise ValueError("support only hdf5 or npy format.")
        
        # del WavLM_feat
        ############ mel extraction ###############
        # mel = logmelfilterbank(
        #     x,
        #     sampling_rate=sampling_rate,
        #     hop_size=hop_size,
        #     fft_size=config["fft_size"],
        #     win_length=config["win_length"],
        #     window=config["window"],
        #     num_mels=config["num_mels"],
        #     fmin=config["fmin"], ## 40
        #     fmax=config["fmax"], ## 700
        #     # keep compatibility
        #     log_base=config.get("log_base", 10.0),
        # )
        # print(mel.shape)
        # print(type(mel))
        # ipdb.set_trace()
        # # make sure the audio length and feature length are matched
        # audio = np.pad(audio, (0, config["fft_size"]), mode="reflect")
        # audio = audio[: len(mel) * config["hop_size"]]
        # assert len(mel) * config["hop_size"] == len(audio)

        # # apply global gain
        # if config["global_gain_scale"] > 0.0:
        #     audio *= config["global_gain_scale"]
        # if np.abs(audio).max() >= 1.0:
        #     logging.warn(
        #         f"{utt_id} causes clipping. "
        #         f"it is better to re-consider global gain scale."
        #     )
        #     continue
        # print("mel")
        # print(type(mel))
        # print(mel.shape)
        # # save
        # if config["format"] == "hdf5":
        #     write_hdf5(
        #         os.path.join(args.dumpdir, f"{utt_id}.h5"),
        #         "wave",
        #         audio.astype(np.float32),
        #     )
        #     write_hdf5(
        #         os.path.join(args.dumpdir, f"{utt_id}.h5"),
        #         "feats",
        #         mel.astype(np.float32),
        #     )
        # elif config["format"] == "npy":
        #     np.save(
        #         os.path.join(args.dumpdir, f"{utt_id}-wave.npy"),
        #         audio.astype(np.float32),
        #         allow_pickle=False,
        #     )
        #     np.save(
        #         os.path.join(args.dumpdir, f"{utt_id}-feats.npy"),
        #         mel.astype(np.float32),
        #         allow_pickle=False,
        #     )
        # else:
        #     raise ValueError("support only hdf5 or npy format.")

        ############# ap MCC extraction ###############
        # x = np.array(x, dtype=np.float32)

        # f0, spc, ap = feature_extractor.analyze(x)
        # codeap = feature_extractor.codeap()
        # mcep = feature_extractor.mcep(dim=config['mcep_dim'], alpha=config['mcep_alpha'])
        # mcep = mcep[:,1:]
        # mcc_ap = np.concatenate([mcep, ap], axis=1)
        # print("mcep ap")
        # print(mcc_ap)
        # print(mcc_ap.shape)


        # # # load converted feature
        # # filepath_np = os.path.join("/home/bioasp/Desktop/temp_hdf5/", f"{utt_id}.npy")
        # # finetune_mcc_ap = np.load(filepath_np)
        # # print("finetune_mcc_ap")
        # # print(finetune_mcc_ap)
        # # print(finetune_mcc_ap.shape)
        # # # assert finetune_mcc_ap.shape == mcc_ap.shape
        # # finetune_el_PWG = False

        # # if finetune_el_PWG == True :
        # #     mcc_ap_hdf5 = finetune_mcc_ap
        # #     print("fintune el data")
        # # else :
        # #     mcc_ap_hdf5 = mcc_ap
        # #     print("normal training data")
        # ###### need if normal progress
        # mcc_ap_hdf5 = mcc_ap
        # ###### need if normal progress

        # # make sure the audio length and feature length are matched
        # audio = np.pad(audio, (0, config["fft_size"]), mode="reflect")
        # audio = audio[: len(mcc_ap) * config["hop_size"]]
        # assert len(mcc_ap) * config["hop_size"] == len(audio)

        # # apply global gain
        # if config["global_gain_scale"] > 0.0:
        #     audio *= config["global_gain_scale"]
        # if np.abs(audio).max() >= 1.0:
        #     logging.warn(
        #         f"{utt_id} causes clipping. "
        #         f"it is better to re-consider global gain scale."
        #     )
        #     continue

        # # save
        # if config["format"] == "hdf5":
        #     write_hdf5(
        #         os.path.join(args.dumpdir, f"{utt_id}.h5"),
        #         "wave",
        #         audio.astype(np.float32),
        #     )
        #     write_hdf5(
        #         os.path.join(args.dumpdir, f"{utt_id}.h5"),
        #         "feats",
        #         mcc_ap_hdf5.astype(np.float32),
        #     )
        # elif config["format"] == "npy":
        #     np.save(
        #         os.path.join(args.dumpdir, f"{utt_id}-wave.npy"),
        #         audio.astype(np.float32),
        #         allow_pickle=False,
        #     )
        #     np.save(
        #         os.path.join(args.dumpdir, f"{utt_id}-feats.npy"),
        #         mcc_ap_hdf5.astype(np.float32),
        #         allow_pickle=False,
        #     )
        # else:
        #     raise ValueError("support only hdf5 or npy format.")


if __name__ == "__main__":
    main()

