import math

import torch
import torch.nn.functional as F
from importlib import import_module
import json
import copy
import ipdb
# Load Config.
str_config = "conf/config_roy_el_cdvqvae_vc.json"
with open(str_config) as f:
    data = f.read()
config = json.loads(data)
model_config = config["model_config"]

### load org model ###
model_type = "new_roy_cdvqvae"
org_module = import_module('model.{}'.format(model_type), package=None)
org_MODEL = getattr(org_module, 'Model')
org_model = org_MODEL(model_config)

model_path = "exp/checkpoints_new_el_cdvqvae/06-20_04-43_1000000"
org_model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
print(org_model)
print("---------------------")
org_model.cuda().eval()
# org_decoder = copy.deepcopy(org_model.decoder['mcc'])
# for p1, p2 in zip(org_decoder.parameters(), org_model.decoder['mcc'].parameters()):
#     print(not(torch.equal(p1, p2)))
# org_encoder = copy.deepcopy(org_model.encoder['mcc'])
# roy_encoder = copy.deepcopy(org_model.encoder['mcc'])
org_spk_embeds = copy.deepcopy(org_model.spk_embeds)
for param in org_model.spk_embeds.parameters():
    print(param)


a = 1
b = copy.deepcopy(a)

# assert a == b
# print(a)
# print(b)