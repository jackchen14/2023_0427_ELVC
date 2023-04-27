import torch
import ipdb
stat_path = "/home/4TB_storage/hsinhao_storage/AVEL_align/cdvae-align-feature-sly-align/data/train_nl/stats.pt"
class WavLM_function:
    def statistic(feat, min_value=1e-10):
        stat_dict = dict()
        if 'WavLM' in feat.keys():
            WavLM = feat['WavLM']
            assert WavLM.ndim == 2
            # mel = torch.log(torch.clamp(mel, min=min_value=1e-10))
            WavLM_mean = torch.mean( WavLM, dim=-1).view(1,-1,1).float()
            WavLM_std = torch.std( WavLM, dim=-1).view(1,-1,1).float()
            stat_dict['WavLM_mean'] = WavLM_mean
            stat_dict['WavLM_std'] = WavLM_std
            # print("mel_mean \n")
            # print(mel_mean)
            # print("mel_std \n")
            # print(mel_std)
            # ipdb.set_trace()
        return stat_dict

    def normalize(feat, min_value=1e-10):
        if 'WavLM' in feat.keys():
            feat_stat = torch.load(stat_path)
            # print(feat_stat['WavLM_mean'])
            # print(feat_stat['WavLM_std'])
            print("WavLM normalize")         
            WavLM = feat['WavLM']
            # mel = torch.log(torch.clamp(mel, min=min_value))           
            WavLM = (WavLM - feat_stat['WavLM_mean']) / feat_stat['WavLM_std']
            feat['WavLM'] = WavLM
        return feat

    def denormalize(feat):
        if 'WavLM' in feat.keys():
            feat_stat = torch.load(stat_path)
            wavlm_std = feat_stat['WavLM_std'].cuda().detach()
            wavlm_mean = feat_stat['WavLM_mean'].cuda().detach()#.cpu().detach()
            print("WavLM Denormalize")         
            WavLM = feat['WavLM']
            a = (WavLM * wavlm_std) + wavlm_mean
            feat['WavLM'] = a
        return feat