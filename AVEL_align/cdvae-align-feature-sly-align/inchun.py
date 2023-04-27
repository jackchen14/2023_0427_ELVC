import os
import librosa
import torch
from WavLM_dir.WavLM import WavLM, WavLMConfig

checkpoint = torch.load('/home/4TB_storage/hsinhao_storage/AVEL_align/cdvae-align-feature-sly-align/WavLM-Base.pt')
cfg = WavLMConfig(checkpoint['cfg'])
wavlm_model = WavLM(cfg)
wavlm_model.load_state_dict(checkpoint['model'])
wavlm_model.eval()

# Define the path to the directory containing the audio files
audio_dir = "/home/4TB_storage/hsinhao_storage/AVEL_align/data/SVSNet_dataset_16kHz"

# Define the path to the directory where you want to save the spectrogram files
WavLM_feature_dir = "/home/4TB_storage/hsinhao_storage/AVEL_align/data/SVSNet_to_WavLM"

# Define the parameters for the mel spectrogram computation
# n_fft = 2048
# hop_length = 512
# n_mels = 128

# Loop through all the audio files in the directory
for filename in os.listdir(audio_dir):
    if filename.endswith(".wav"):
        # Load the audio file
        audio_path = os.path.join(audio_dir, filename)
        y, sr = librosa.load(audio_path, sr=None)

        y = torch.from_numpy(y)
        y = y.unsqueeze(0)
        print(y.shape)

        with torch.no_grad():
            # if cfg.normalize:
            #     wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz , wav_input_16khz.shape)
            rep, layer_results = wavlm_model.extract_features(y, output_layer=wavlm_model.cfg.encoder_layers, ret_layer_results=True)[0]
            layer_reps = [x.transpose(0, 1) for x, _ in layer_results]

        # Save the spectrogram as a PyTorch tensor
        WavLM_path = os.path.join(WavLM_feature_dir, filename.replace(".wav", ".pt"))
        torch.save(layer_reps, WavLM_path)

print("sampling_rate = " + str(sr))
print("shape_of_file : ")
print(len(layer_reps))
print(layer_reps[0].shape)


