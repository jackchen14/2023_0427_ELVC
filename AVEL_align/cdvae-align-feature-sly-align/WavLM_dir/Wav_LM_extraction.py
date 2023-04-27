import torch
import numpy as np
from WavLM import WavLM, WavLMConfig

# load the pre-trained checkpoints
checkpoint = torch.load('/home/bioasp/Downloads/WavLM-Base.pt')
# for a in checkpoint :
#     for b in checkpoint[a]:
#         print(b , checkpoint[a][b])
#         if b == 'mask_emb' :
#             break
        
cfg = WavLMConfig(checkpoint['cfg'])
model = WavLM(cfg)
model.load_state_dict(checkpoint['model'])
model.eval()

# extract the representation of last layer
wav_input_16khz = torch.randn(1,50001)
rep = model.extract_features(wav_input_16khz)[0]
print(rep.shape)

# # extract the representation of each layer
# wav_input_16khz = torch.randn(1,50001)
# rep, layer_results = model.extract_features(wav_input_16khz, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
# layer_reps = [x.transpose(0, 1) for x, _ in layer_results]

# for i in range(len(layer_reps)) :
#     print (i , layer_reps[i].shape)