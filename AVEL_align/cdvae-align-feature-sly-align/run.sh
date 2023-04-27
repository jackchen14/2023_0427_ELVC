#!/bin/bash

stage=0
stop_stage=0
step=0
gpu=0
### make new test list
make_trial=true

conf_dir=conf
model

TMSV_root='/home/4TB_storage/hsinhao_storage/AVEL_align/data/TMSV/wav'
tmhint_nl_root='/home/4TB_storage/hsinhao_storage/AVEL_align/data/NL01/wav'
tmhint_el_root='/home/4TB_storage/hsinhao_storage/AVEL_align/data/EL01/wav'
# tmhint_nl_root='/home/bioasp/Downloads/NL01a2_audio/NL01'
# tmhint_el_root='/home/bioasp/Downloads/nEL01a2_audio/EL01'

src_spk=NL01
tgt_spk=EL01

if [ -f ./path.sh ]; then . ./path.sh; fi
. utils/parse_options.sh || exit 1;


# Generate list of Speakers
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo -e "stage 0: data list generation"
    python data_filter.py $tmhint_el_root $tmhint_nl_root
    python local/generate_TMHINT_NL_list.py -d $TMSV_root -l data
    utils/utt2spk_to_spk2utt.pl data/tmhint_train/utt2spk  data/tmhint_train/spk2utt
    utils/utt2spk_to_spk2utt.pl data/tmhint_test/utt2spk > data/tmhint_test/spk2utt

    python local/generate_TMHINT_EL_N_list.py -d $tmhint_nl_root -l data
    utils/utt2spk_to_spk2utt.pl data/tmhint_nl_train/utt2spk > data/tmhint_nl_train/spk2utt
    utils/utt2spk_to_spk2utt.pl data/tmhint_nl_test/utt2spk > data/tmhint_nl_test/spk2utt

    python local/generate_TMHINT_EL_E_list.py -d $tmhint_el_root -l data
    utils/utt2spk_to_spk2utt.pl data/tmhint_el_train/utt2spk > data/tmhint_el_train/spk2utt
    utils/utt2spk_to_spk2utt.pl data/tmhint_el_test/utt2spk > data/tmhint_el_test/spk2utt    

    utils/combine_data.sh data/train_nl data/tmhint_train data/tmhint_nl_train
    python local/make_spk_id.py data/train_nl

    utils/combine_data.sh data/test_nl data/tmhint_test data/tmhint_nl_test
    cp data/train_nl/spk2spk_id data/test_nl/

    cp data/train_nl/spk2spk_id data/tmhint_nl_train/
    cp data/train_nl/spk2spk_id data/tmhint_nl_test/
    python local/make_spk_id.py data/tmhint_nl_train

    cp data/train_nl/spk2spk_id data/tmhint_el_train/
    cp data/train_nl/spk2spk_id data/tmhint_el_test/
    echo -e "stage 0: data list generation done!"
fi

# Preprocessing
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo -e "stage 1: TMSV feature extraction"
    mkdir -p features
    python feature_extraction.py -c conf/config_feature_WavLM_mel.json \
        -T data/train_nl -F features/train_nl \
        -K "mel- " -W "WavLM_Yes"

    python feature_extraction.py -c conf/config_feature_WavLM_mel.json \
        -T data/test_nl -F features/test_nl \
        -K "mel- " -W "WavLM_Yes"        

    # Statistic parameters for normalization
    python feature_statistic.py -c conf/config_feature_WavLM_mel.json \
        -T data/train_nl -S data/train_nl/stats.pt \
        -K "mel- " # -Wavlm

    # Normalize features
    mkdir -p data/train_nl_cmvn
    cp -r data/train_nl/* data/train_nl_cmvn
    python feature_normalization.py -c conf/config_feature_WavLM_mel.json \
        -T data/train_nl_cmvn -F features/train_nl_cmvn \
        -S data/train_nl/stats.pt \
        -K "mel-WavLM"

    mkdir -p data/test_nl_cmvn
    cp -r data/test_nl/* data/test_nl_cmvn
    python feature_normalization.py -c conf/config_feature_WavLM_mel.json \
        -T data/test_nl_cmvn -F features/test_nl_cmvn \
        -S data/train_nl/stats.pt \
        -K "mel-WavLM"
    
    echo -e "stage 1: TMSV feature extraction done!"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo -e "stage 2: ELVC pair feature extraction"

    python feature_extraction.py -c conf/config_feature_WavLM_mel.json \
        -T data/tmhint_el_train -F features/tmhint_el_train \
        -K "mel- " -W "WavLM_Yes"

    python feature_extraction.py -c conf/config_feature_WavLM_mel.json \
        -T data/tmhint_nl_train -F features/tmhint_nl_train \
        -K "mel- " -W "WavLM_Yes"

    python feature_extraction.py -c conf/config_feature_WavLM_mel.json \
        -T data/tmhint_el_test -F features/tmhint_el_test \
        -K "mel- " -W "WavLM_Yes"

    # Normalize features
    mkdir -p data/tmhint_el_train_cmvn
    cp -r data/tmhint_el_train/* data/tmhint_el_train_cmvn
    python feature_normalization.py -c conf/config_feature_WavLM_mel.json \
        -T data/tmhint_el_train_cmvn -F features/tmhint_el_train_cmvn \
        -S data/train_nl/stats.pt \
        -K "mel-WavLM"

    mkdir -p data/tmhint_nl_train_cmvn
    cp -r data/tmhint_nl_train/* data/tmhint_nl_train_cmvn
    python feature_normalization.py -c conf/config_feature_WavLM_mel.json \
        -T data/tmhint_nl_train_cmvn -F features/tmhint_nl_train_cmvn \
        -S data/train_nl/stats.pt \
        -K "mel-WavLM"

    mkdir -p data/tmhint_el_test_cmvn
    cp -r data/tmhint_el_test/* data/tmhint_el_test_cmvn
    python feature_normalization.py -c conf/config_feature_WavLM_mel.json \
        -T data/tmhint_el_test_cmvn -F features/tmhint_el_test_cmvn \
        -S data/train_nl/stats.pt \
        -K "mel-WavLM"
    
    echo -e "stage 2: ELVC pair feature extraction done!"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then

    echo -e "stage 3: CMVN calculation"
    
    mkdir -p data/tmhint_elnl_train_cmvn
    cp -r data/tmhint_nl_train_cmvn/* data/tmhint_elnl_train_cmvn
    python local/combine_data_ark.py \
        data/tmhint_elnl_train_cmvn features/tmhint_elnl_train_cmvn \
        "el:data/tmhint_el_train_cmvn,nl:data/tmhint_nl_train_cmvn" \
        -K "mcc-sp-mel"

    python feature_alignment_org_utt.py \
        data/tmhint_elnl_train_cmvn \
        -K "nl_mcc:el_mcc" -U "2"

    echo -e "stage 3: CMVN calculation done!"

fi

### syl wavlm dtw ###
if [ ${stage} -le 31 ] && [ ${stop_stage} -ge 31 ]; then

    echo -e "stage 31: CMVN calculation"
    
    mkdir -p data/tmhint_elnl_train_cmvn
    cp -r data/tmhint_nl_train_cmvn/* data/tmhint_elnl_train_cmvn
    python local/combine_data_ark.py \
        data/tmhint_elnl_train_cmvn features/tmhint_elnl_train_cmvn \
        "el:data/tmhint_el_train_cmvn,nl:data/tmhint_nl_train_cmvn" \
        -K "mcc-sp-mel-WavLM"

    python new_feature_alignment.py \
        data/tmhint_elnl_train_cmvn \
        -K "nl_WavLM:el_WavLM" -U "2"

    echo -e "stage 31: CMVN calculation done!"

fi

### syl mcc dtw ###
if [ ${stage} -le 32 ] && [ ${stop_stage} -ge 32 ]; then

    echo -e "stage 31: CMVN calculation"
    
    mkdir -p data/tmhint_elnl_train_cmvn
    cp -r data/tmhint_nl_train_cmvn/* data/tmhint_elnl_train_cmvn
    python local/combine_data_ark.py \
        data/tmhint_elnl_train_cmvn features/tmhint_elnl_train_cmvn \
        "el:data/tmhint_el_train_cmvn,nl:data/tmhint_nl_train_cmvn" \
        -K "mcc-sp-mel"

    python feature_alignment_org_syl.py \
        data/tmhint_elnl_train_cmvn \
        -K "nl_mel:el_mel" -U "2"

    echo -e "stage 31: CMVN calculation done!"

fi
### utt level dtw
if [ ${stage} -le 33 ] && [ ${stop_stage} -ge 33 ]; then

    echo -e "stage 33: CMVN calculation"
    
    mkdir -p data/tmhint_elnl_train_cmvn
    cp -r data/tmhint_nl_train_cmvn/* data/tmhint_elnl_train_cmvn
    python local/combine_data_ark.py \
        data/tmhint_elnl_train_cmvn features/tmhint_elnl_train_cmvn \
        "el:data/tmhint_el_train_cmvn,nl:data/tmhint_nl_train_cmvn" \
        -K "mcc-sp-mel"

    python feature_alignment_org_utt.py \
        data/tmhint_elnl_train_cmvn \
        -K "nl_mcc:el_mcc" -U "2"

    echo -e "stage 33: CMVN calculation done!"

fi

if [ ${stage} -le 34 ] && [ ${stop_stage} -ge 34 ]; then

    echo -e "stage 31: CMVN calculation"
    
    mkdir -p data/tmhint_elnl_train_cmvn
    cp -r data/tmhint_nl_train_cmvn/* data/tmhint_elnl_train_cmvn
    python local/combine_data_ark.py \
        data/tmhint_elnl_train_cmvn features/tmhint_elnl_train_cmvn \
        "el:data/tmhint_el_train_cmvn,nl:data/tmhint_nl_train_cmvn" \
        -K "mel-WavLM"

    python feature_alignment_mel_wavlm.py \
        data/tmhint_elnl_train_cmvn \
        -K "nl_WavLM:el_WavLM" -U "2"

    echo -e "stage 31: CMVN calculation done!"

fi
# Training
### init cdvqvae vc for NL speech
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then

    echo -e "${YW}stage 4: CDVAE pre-training${NC}"
    python train.py -g $gpu -c conf/config_cdpvqvae_vc.json \
        -T data/train_nl_cmvn -o exp/checkpoints_CDPJ_VQVAE
    
    ### change lr
    # python train.py -g $gpu -c conf/config_cdpvqvae_vc.json \
    #     -T data/train_nl_cmvn -o exp/checkpoints_CDPJ_VQVAE_lr2e-4 -lr 0.0002
    # python train.py -g $gpu -c conf/config_cdpvqvae_vc.json \
    #     -T data/train_nl_cmvn -o exp/checkpoints_CDPJ_VQVAE_lr5e-5 -lr 0.00005

    echo -e "${YW}stage 4: CDVAE pre-training done!${NC}"
fi



### init cdvqvae vc for NL speech
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then

    if [ $step == 9.0 ]; then
        echo -e "${YW}stage 9: CDVAE pre-training${NC}"
        python train.py -g $gpu -c conf/config_new_cdvqvae_vc.json \
            -T data/train_nl_cmvn -o exp/checkpoints_new_cdvqvae \

    fi

    if [ $step == 9.1 ]; then
        echo -e "${YW}stage 9: roy_CDVAE (no vq) pre-training${NC}"
        python train.py -g $gpu -c conf/config_stage1_roy_cdvqvae_vc.json \
            -T data/train_nl_cmvn -o exp/checkpoints_stage1_roy \

    fi

    if [ $step == 9.2 ]; then
        echo -e "${YW}stage 9: mel_roy_CDVAE (no vq) pre-training${NC}"
        python train.py -g $gpu -c conf/config_stage1_mel_roy_cdvqvae_vc.json \
            -T data/train_nl_cmvn -o exp/checkpoints_stage1_mel_roy \

    fi

    if [ $step == 9.3 ]; then
        echo -e "${YW}stage 9: mel_wavlm_roy_CDVAE (no vq) pre-training${NC}"
        python train.py -g $gpu -c conf/config_stage1_mel_wavlm_roy_cdvqvae_vc.json \
            -T data/train_nl_cmvn -o exp/checkpoints_stage1_mel_wavlm_roy \

    fi

    if [ $step == 9.4 ]; then
        echo -e "${YW}stage 9: CDVAE-GAN pre-training${NC}"
        python train.py -g $gpu -c conf/config_cdvqvawgan_vc.json \
            -T data/train_nl_cmvn -o exp/checkpoints_new_cdvqvae_GAN
    fi

    if [ $step == 9.5 ]; then
        echo -e "${YW}stage 9: CDVAE-GAN pre-training${NC}"
        python train.py -g $gpu -c conf/config_stage1_wavlmXwavlm_2023_roy_cdvqvae_vc.json \
            -T data/train_nl_cmvn -o exp/checkpoints_wavlmXwavlm_2023
    fi

    if [ $step == 9.6 ]; then
        echo -e "${YW}stage 9: VAE-WavLM pre-training${NC}"
        python train.py -g $gpu -c conf/config_stage1_vae_wavlm_2023.json \
            -T data/train_nl_cmvn -o exp/checkpoints_stage1_vae_wavlm_2023.py
    fi
    ### change lr
    # python train.py -g $gpu -c conf/config_cdpvqvae_vc.json \
    #     -T data/train_nl_cmvn -o exp/checkpoints_CDPJ_VQVAE_lr2e-4 -lr 0.0002
    # python train.py -g $gpu -c conf/config_cdpvqvae_vc.json \
    #     -T data/train_nl_cmvn -o exp/checkpoints_CDPJ_VQVAE_lr5e-5 -lr 0.00005

    echo -e "${YW}stage 9: CDVAE pre-training done!${NC}"
fi
## new ### cdvae nl pretrain + cdvae el train

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then

    if [ $step == 10.0 ]; then
        echo -e "${YW}stage 10: CDVAE el_training${NC}"
        python train.py -g $gpu -c conf/config_new_el_cdvqvae_vc.json \
            -T data/tmhint_elnl_train_cmvn -o exp/checkpoints_new_el_cdvqvae \
        ########## checkpoints_new_cdvqvae/03-08_04-07_600000
        ### change lr
        # python train.py -g $gpu -c conf/config_cdpvqvae_vc.json \
        #     -T data/train_nl_cmvn -o exp/checkpoints_CDPJ_VQVAE_lr2e-4 -lr 0.0002
        # python train.py -g $gpu -c conf/config_cdpvqvae_vc.json \
        #     -T data/train_nl_cmvn -o exp/checkpoints_CDPJ_VQVAE_lr5e-5 -lr 0.00005
        echo -e "${YW}stage 10.1: CDVAE el_training done!${NC}"
    fi

    if [ $step == 10.1 ]; then
    echo -e "${YW}stage 10: CDVAE el_training${NC}"
        python train.py -g $gpu -c conf/config_roy_el_cdvqvae_vc.json \
            -T data/tmhint_elnl_train_cmvn -o exp/checkpoints_new_el_cdvqvae \
            # -p exp/checkpoints_new_el_cdvqvae/06-09_08-29_780000
        ########## checkpoints_new_cdvqvae/03-08_04-07_600000
        ### change lr
        # python train.py -g $gpu -c conf/config_cdpvqvae_vc.json \
        #     -T data/train_nl_cmvn -o exp/checkpoints_CDPJ_VQVAE_lr2e-4 -lr 0.0002
        # python train.py -g $gpu -c conf/config_cdpvqvae_vc.json \
        #     -T data/train_nl_cmvn -o exp/checkpoints_CDPJ_VQVAE_lr5e-5 -lr 0.00005
        echo -e "${YW}stage 10.1: Roy CDVAE el_training done!${NC}"
    fi

    if [ $step == 10.2 ]; then
    echo -e "${YW}stage 10: roy stage2 el_training${NC}"
        python train.py -g $gpu -c conf/config_stage2_mel_roy_cdvqvae_vc.json \
            -T data/tmhint_elnl_train_cmvn -o exp/checkpoints_stage2_mel_roy \
            # -p exp/checkpoints_new_el_cdvqvae/06-09_08-29_780000
        ########## checkpoints_new_cdvqvae/03-08_04-07_600000
        ### change lr
        # python train.py -g $gpu -c conf/config_cdpvqvae_vc.json \
        #     -T data/train_nl_cmvn -o exp/checkpoints_CDPJ_VQVAE_lr2e-4 -lr 0.0002
        # python train.py -g $gpu -c conf/config_cdpvqvae_vc.json \
        #     -T data/train_nl_cmvn -o exp/checkpoints_CDPJ_VQVAE_lr5e-5 -lr 0.00005
        echo -e "${YW}stage 10.2: roy stage2 el_training done!${NC}"
    fi

    if [ $step == 10.3 ]; then
    echo -e "${YW}stage 10: roy stage2 el_training${NC}"
        python train.py -g $gpu -c conf/config_stage2_mel_wavlm_roy_cdvqvae_vc.json \
            -T data/tmhint_elnl_train_cmvn -o exp/checkpoints_stage2_mel_wavlm_roy \
            # -p exp/checkpoints_new_el_cdvqvae/06-09_08-29_780000
        ########## checkpoints_new_cdvqvae/03-08_04-07_600000
        ### change lr
        # python train.py -g $gpu -c conf/config_cdpvqvae_vc.json \
        #     -T data/train_nl_cmvn -o exp/checkpoints_CDPJ_VQVAE_lr2e-4 -lr 0.0002
        # python train.py -g $gpu -c conf/config_cdpvqvae_vc.json \
        #     -T data/train_nl_cmvn -o exp/checkpoints_CDPJ_VQVAE_lr5e-5 -lr 0.00005
        echo -e "${YW}stage 10.3: roy wavlm stage2 el_training done!${NC}"
    fi

    if [ $step == 10.4 ]; then
        echo -e "${YW}stage 10: roy stage2 el_training${NC}"
        python train.py -g $gpu -c conf/config_stage2_wavlm_main_roy_cdvqvae_vc.json \
            -T data/tmhint_elnl_train_cmvn -o exp/checkpoints_stage2_wavlm_main_roy \
            # -p exp/checkpoints_new_el_cdvqvae/06-09_08-29_780000
        ########## checkpoints_new_cdvqvae/03-08_04-07_600000
        ### change lr
        # python train.py -g $gpu -c conf/config_cdpvqvae_vc.json \
        #     -T data/train_nl_cmvn -o exp/checkpoints_CDPJ_VQVAE_lr2e-4 -lr 0.0002
        # python train.py -g $gpu -c conf/config_cdpvqvae_vc.json \
        #     -T data/train_nl_cmvn -o exp/checkpoints_CDPJ_VQVAE_lr5e-5 -lr 0.00005
        echo -e "${YW}stage 10.3: roy wavlm stage2 el_training done!${NC}"
    fi

    if [ $step == 10.5 ]; then
        echo -e "${YW}stage 10.5 : 11_16 cross-domain start ${NC}"
        python train.py -g $gpu -c conf/config_stage2_cross_domain_1116.json \
            -T data/tmhint_elnl_train_cmvn -o exp/checkpoints_stage2_cross_domain \
            -p exp/checkpoints_stage1_mel_wavlm_roy/07-29_08-54_600000
        ########## checkpoints_new_cdvqvae/03-08_04-07_600000
        ### change lr
        # python train.py -g $gpu -c conf/config_cdpvqvae_vc.json \
        #     -T data/train_nl_cmvn -o exp/checkpoints_CDPJ_VQVAE_lr2e-4 -lr 0.0002
        # python train.py -g $gpu -c conf/config_cdpvqvae_vc.json \
        #     -T data/train_nl_cmvn -o exp/checkpoints_CDPJ_VQVAE_lr5e-5 -lr 0.00005
        echo -e "${YW}stage 10.5: 11_16 cross-domain done ${NC}"
    fi


    if [ $step == 10.6 ]; then
        echo -e "${YW}stage 10.6:WavLM_only_${NC}"
        python train.py -g $gpu -c conf/config_stage2_vae_wavlm_2023.json \
            -T data/tmhint_elnl_train_cmvn -o exp/checkpoints_stage2_vae_wavlm_2023 \
            # -p exp/checkpoints_new_el_cdvqvae/06-09_08-29_780000
        ########## checkpoints_new_cdvqvae/03-08_04-07_600000
        ### change lr
        # python train.py -g $gpu -c conf/config_cdpvqvae_vc.json \
        #     -T data/train_nl_cmvn -o exp/checkpoints_CDPJ_VQVAE_lr2e-4 -lr 0.0002
        # python train.py -g $gpu -c conf/config_cdpvqvae_vc.json \
        #     -T data/train_nl_cmvn -o exp/checkpoints_CDPJ_VQVAE_lr5e-5 -lr 0.00005
        echo -e "${YW}stage 10.3: roy wavlm stage2 el_training done!${NC}"
    fi


fi

if [ ${stage} -le 699 ] && [ ${stop_stage} -ge 699 ]; then

    echo -e "${YW}stage 9: CDVAE pre-training${NC}"
    python train.py -g $gpu -c conf/config_cdvqvae_vc.json \
        -T data/train_nl_cmvn -o exp/checkpoints_new_cdpvqvae
    
    ### change lr
    # python train.py -g $gpu -c conf/config_cdpvqvae_vc.json \
    #     -T data/train_nl_cmvn -o exp/checkpoints_CDPJ_VQVAE_lr2e-4 -lr 0.0002
    # python train.py -g $gpu -c conf/config_cdpvqvae_vc.json \
    #     -T data/train_nl_cmvn -o exp/checkpoints_CDPJ_VQVAE_lr5e-5 -lr 0.00005

    echo -e "${YW}stage 9: CDVAE pre-training done!${NC}"
fi



### train decoder for NL speech
### setting pre-train model from stage 4
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo -e "stage MEL decoder"
    
    ### pre-trained model=VQVAE
    ### baseline decoder
    if [ $step == 5.1 ]; then
        echo -e "step: ${step} base MEL decoder"
        python train.py -g $gpu -c conf/config_cdpmeldec_vc.json \
            -T data/train_nl_cmvn -o exp/checkpoints_CDPJ_MELDEC \
            -p exp/checkpoints_CDPJ_VQVAE/10-29_10-19_600000
    fi
    ### TFM decoder
    if [ $step == 5.2 ]; then
        echo -e "step: ${step} TFM MEL decoder"
        python train.py -g $gpu -c conf/config_cdpmeltfm_vc.json \
            -T data/train_nl_cmvn -o exp/checkpoints_CDPJ_MELTFM \
            -p exp/checkpoints_CDPJ_VQVAE/10-19_04-14_600000
    fi

    echo -e "stage 5: training MEL decoder done!"

fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo -e "stage 7: self reconstruction"

    if [ $step == 6.0 ]; then
        ### original pre-trained model
        src_spk=NL01
        tgt_spk=NL01
        model=exp/checkpoints_new_cdvqvae/03-08_04-07_600000
        echo -e "current model: ${model}"

        [ $make_trial == 'true' -o ! -f data/test/trials ] && \
            python local/make_trials.py data/test_nl_cmvn -s $src_spk -t $tgt_spk -n 75

        python inference.py -c conf/config_new_cdvqvae_vc.json \
            -d data/test_nl_cmvn -t data/test_nl_cmvn/trials \
            -ow exp/result_wav/${src_spk}_${tgt_spk}_new_cdvae_01_12 \
            -K "mcc-mcc" \
            -m ${model} -g $gpu
    fi

    if [ $step == 6.1 ]; then
        ### original pre-trained model
        src_spk=NL01
        tgt_spk=NL01
        model=exp/checkpoints_new_cdvqvae/03-08_04-07_600000
        echo -e "current model: ${model}"

        [ $make_trial == 'true' -o ! -f data/test/trials ] && \
            python local/make_trials.py data/train_nl_cmvn -s $src_spk -t $tgt_spk -n 224

        python inference.py -c conf/config_new_cdvqvae_vc.json \
            -d data/train_nl_cmvn -t data/train_nl_cmvn/trials \
            -ow exp/result_wav/${src_spk}_${tgt_spk}_new_cdvae_01_12 \
            -K "mcc-mcc" \
            -m ${model} -g $gpu
    fi

    if [ $step == 6.2 ]; then
        ### TFM mel-decoder with pre-trained CDPJ-VQVAE model
        src_spk=NL01
        tgt_spk=NL01
        model=exp/checkpoints_CDPJ_MELTFM/09-01_13-44_500000
        echo -e "current model: ${model}"

        [ $make_trial == 'true' -o ! -f data/test/trials ] && \
            python local/make_trials.py data/test_nl_cmvn -s $src_spk -t $tgt_spk -n 10

        python inference.py -c conf/config_cdpmeltfm_vc.json \
            -d data/test_nl_cmvn -t data/test_nl_cmvn/trials \
            -ow exp/result_wav/${src_spk}_${tgt_spk}_cdpjmel_tfm \
            -K "mcc-mel" \
            -m ${model} -g $gpu
    fi

    echo -e "stage 7: self reconstruction done!" 
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo -e "stage 7: training EL encoder"
    
    if [ $step == 7.1 ]; then
        echo -e "step: ${step} base EL encoder 1"
        python train.py -g $gpu -c conf/config_cdpelcvt_vc.json \
            -T data/tmhint_elnl_train_cmvn -o exp/checkpoints_CDP_ELCVT_MC \
            -p exp/checkpoints_CDPJ_MELDEC/10-30_23-53_1000000
    fi
    if [ $step == 7.2 ]; then
        echo -e "step: ${step} base EL encoder mcsp"
        python train.py -g $gpu -c conf/config_cdpelcvt_vc_mcsp.json \
            -T data/tmhint_elnl_train_cmvn -o exp/checkpoints_CDP_ELCVT_MCSP \
            -p exp/checkpoints_CDPJ_MELDEC/10-19_20-54_1000000
    fi
    if [ $step == 7.3 ]; then
        echo -e "step: ${step} base EL encoder 2"
        python train.py -g $gpu -c conf/config_cdpelcvt2_vc.json \
            -T data/tmhint_elnl_train_cmvn -o exp/checkpoints_CDP_ELCVT2 \
            -p exp/checkpoints_CDPJ_MELDEC/10-19_20-54_1000000
    fi
    if [ $step == 7.4 ]; then
        echo -e "step: ${step} base EL encoder 3"
        python train.py -g $gpu -c conf/config_cdpelcvt3_vc.json \
            -T data/tmhint_elnl_train_cmvn -o exp/checkpoints_CDPJ_ELCVT3 \
            -p exp/checkpoints_CDPJ_MELDEC/10-19_20-54_1000000
    fi
    if [ $step == 7.5 ]; then
        echo -e "step: ${step} base TFM EL decoder"
        python train.py -g $gpu -c conf/config_cdpelcvt4_vc.json \
            -T data/tmhint_elnl_train_cmvn -o exp/checkpoints_CDPJ_ELCVT_TFM \
            -p exp/checkpoints_CDPJ_MELTFM/10-19_20-54_1000000
    fi

    echo -e "stage 7: training EL encoder done!"

fi


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo -e "stage 8: converted EL speech wavetable synthesis"

    if [ $step == 8.0 ]; then
        src_spk=EL01
        tgt_spk=NL01
        model=exp/checkpoints_stage2_wavlm_main_roy/08-03_22-44_400000
        # model=exp/checkpoints_stage2_mel_wavlm_roy/07-29_19-47_400000
        
        [ $make_trial == 'true' -o ! -f data/test/trials ] && \
            python local/make_trials.py data/tmhint_el_test_cmvn -s $src_spk -t $tgt_spk -n 80
        # conf/config_stage2_mel_wavlm_roy_cdvqvae_vc.json\
        echo -e "model: ${model}"
        python inference.py -c conf/config_stage2_wavlm_main_roy_cdvqvae_vc.json\
            -d data/tmhint_el_test_cmvn -t data/tmhint_el_test_cmvn/trials \
            -ow exp/result_wav/${src_spk}_${tgt_spk}_stage2_concate_feat_cdvqvae \
            -K "WavLM-WavLM" \
            -m ${model} -g $gpu
    fi

    if [ $step == 8.1 ]; then
        src_spk=EL01
        tgt_spk=NL01
        # model=exp/checkpoints_stage2_wavlm_main_roy/08-03_22-44_400000
        model=exp/checkpoints_stage2_mel_wavlm_roy/07-29_19-47_400000
        tmhint_elnl_train_cmvn
        [ $make_trial == 'true' -o ! -f data/test/trials ] && \
            python local/make_trials.py data/tmhint_el_train_cmvn -s $src_spk -t $tgt_spk -n 240
        
        echo -e "model: ${model}"
        python inference.py -c conf/config_stage2_mel_wavlm_roy_cdvqvae_vc.json\
            -d data/tmhint_el_train_cmvn -t data/tmhint_el_train_cmvn/trials \
            -ow exp/result_wav/${src_spk}_${tgt_spk}_stage2_concate_feat_cdvqvae \
            -K "mel-mel" \
            -m ${model} -g $gpu
    fi

    if [ $step == 8.2 ]; then
        src_spk=EL01
        tgt_spk=NL01
        model=exp/checkpoints_CDP_ELCVT_MC/10-23_12-14_1000000

        [ $make_trial == 'true' -o ! -f data/test/trials ] && \
            python local/make_trials.py data/tmhint_el_test_cmvn -s $src_spk -t $tgt_spk -n 10
        
        echo -e "model: ${model}"
        python inference.py -c conf/config_cdpelcvt_vc.json \
            -d data/tmhint_el_test_cmvn -t data/tmhint_el_test_cmvn/trials \
            -ow exp/result_wav/${src_spk}_${tgt_spk}_cdpjmel_el_tfm \
            -K "mcc-mel" \
            -m ${model} -g $gpu
    fi

    if [ $step == 8.3 ]; then
        src_spk=EL01
        tgt_spk=NL01
        model=exp/checkpoints_CDP_ELCVT_MC/01-10_19-26_1000000

        [ $make_trial == 'true' -o ! -f data/test/trials ] && \
            python local/make_trials.py data/tmhint_el_test_cmvn -s $src_spk -t $tgt_spk -n 10
        
        echo -e "model: ${model}"
        python inference.py -c conf/config_new_el_cdvqvae_vc.json \
            -d data/tmhint_el_test_cmvn -t data/tmhint_el_test_cmvn/trials \
            -ow exp/result_wav/${src_spk}_${tgt_spk}_letsgo_el \
            -K "mcc-mcc" \
            -m ${model} -g $gpu
    fi

    if [ $step == 8.4 ]; then
        src_spk=NL01
        tgt_spk=NL01
        # model=exp/checkpoints_stage2_wavlm_main_roy/08-03_22-44_400000
        model=exp/checkpoints_stage2_mel_wavlm_roy/07-29_19-47_400000

        [ $make_trial == 'true' -o ! -f data/test/trials ] && \
            python local/make_trials.py data/tmhint_elnl_train_cmvn -s $src_spk -t $tgt_spk -n 240
        
        # config_stage2_wavlm_main_roy_cdvqvae_vc.json \
        echo -e "model: ${model}"
        python inference.py -c conf/config_stage2_mel_wavlm_roy_cdvqvae_vc.json \
            -d data/tmhint_elnl_train_cmvn -t data/tmhint_elnl_train_cmvn/trials \
            -ow exp/result_wav/${src_spk}_${tgt_spk}_test_aligned_wavlm \
            -K "mel-mel" \
            -m ${model} -g $gpu
    fi

    if [ $step == 8.5 ]; then
        src_spk=EL01
        tgt_spk=NL01
        
        model=exp/checkpoints_stage2_mel_roy/10-12_16-32_400000
        # model=exp/checkpoints_stage2_mel_wavlm_roy/07-29_19-47_400000
        
        [ $make_trial == 'true' -o ! -f data/test/trials ] && \
            python local/make_trials.py data/tmhint_el_test_cmvn -s $src_spk -t $tgt_spk -n 80
        # conf/config_stage2_mel_wavlm_roy_cdvqvae_vc.json\
        echo -e "model: ${model}"
        python inference.py -c conf/config_stage2_mel_roy_cdvqvae_vc.json\
            -d data/tmhint_el_test_cmvn -t data/tmhint_el_test_cmvn/trials \
            -ow exp/result_wav/${src_spk}_${tgt_spk}_stage2_mel_roy_dtw_Mel \
            -K "mel-mel" \
            -m ${model} -g $gpu
    fi

    if [ $step == 8.6 ]; then
        src_spk=EL01
        tgt_spk=NL01

        model=exp/checkpoints_stage2_cross_domain/11-23_22-32_400000
        # model=exp/checkpoints_stage2_mel_wavlm_roy/07-29_19-47_400000
        
        [ $make_trial == 'true' -o ! -f data/test/trials ] && \
            python local/make_trials.py data/tmhint_el_test_cmvn -s $src_spk -t $tgt_spk -n 80
        # conf/config_stage2_mel_wavlm_roy_cdvqvae_vc.json\
        echo -e "model: ${model}"
        python inference_cross_domain.py -c conf/config_stage2_cross_domain_1116.json \
            -d data/tmhint_el_test_cmvn -t data/tmhint_el_test_cmvn/trials \
            -ow exp/result_wav/${src_spk}_${tgt_spk}_stage2_cross_domain \
            -K "mel-WavLM-CD_MelWavLM" \
            -m ${model} -g $gpu
    fi

    if [ $step == 8.7 ]; then
        src_spk=EL01
        tgt_spk=NL01

        model=exp/checkpoints_stage2_cross_domain/11-23_22-32_400000
        # model=exp/checkpoints_stage2_mel_wavlm_roy/07-29_19-47_400000
        
        [ $make_trial == 'true' -o ! -f data/test/trials ] && \
            python local/make_trials.py data/tmhint_el_test_cmvn -s $src_spk -t $tgt_spk -n 80
        # conf/config_stage2_mel_wavlm_roy_cdvqvae_vc.json\
        echo -e "model: ${model}"
        python inference.py -c conf/config_stage2_cross_domain_1116.json \
            -d data/tmhint_el_test_cmvn -t data/tmhint_el_test_cmvn/trials \
            -ow exp/result_wav/${src_spk}_${tgt_spk}_WavLM_only_stage2_cross_domain \
            -K "WavLM-WavLM" \
            -m ${model} -g $gpu
    fi

    if [ $step == 8.8 ]; then
        src_spk=EL01
        tgt_spk=NL01

        model=exp/checkpoints_stage2_cross_domain/11-23_22-32_400000
        # model=exp/checkpoints_stage2_mel_wavlm_roy/07-29_19-47_400000
        
        [ $make_trial == 'true' -o ! -f data/test/trials ] && \
            python local/make_trials.py data/tmhint_el_test_cmvn -s $src_spk -t $tgt_spk -n 80
        # conf/config_stage2_mel_wavlm_roy_cdvqvae_vc.json\
        echo -e "model: ${model}"
        python inference.py -c conf/config_stage2_cross_domain_1116.json \
            -d data/tmhint_el_test_cmvn -t data/tmhint_el_test_cmvn/trials \
            -ow exp/result_wav/${src_spk}_${tgt_spk}_mel_only_stage2_cross_domain \
            -K "mel-mel" \
            -m ${model} -g $gpu
    fi

    if [ $step == 8.9 ]; then
        src_spk=EL01
        tgt_spk=NL01

        model=exp/checkpoints_stage2_vae_wavlm_2023/01-23_21-52_400000
        # model=exp/checkpoints_stage2_mel_wavlm_roy/07-29_19-47_400000
        
        [ $make_trial == 'true' -o ! -f data/test/trials ] && \
            python local/make_trials.py data/tmhint_el_test_cmvn -s $src_spk -t $tgt_spk -n 80
        # conf/config_stage2_mel_wavlm_roy_cdvqvae_vc.json\
        echo -e "model: ${model}"
        python inference.py -c conf/config_stage2_vae_wavlm_2023.json \
            -d data/tmhint_el_test_cmvn -t data/tmhint_el_test_cmvn/trials \
            -ow exp/result_wav/stage2_vae_wavlm \
            -K "WavLM-WavLM" \
            -m ${model} -g $gpu
    fi

    echo -e "stage 8: converted EL speech waveform synthesis done!"

fi