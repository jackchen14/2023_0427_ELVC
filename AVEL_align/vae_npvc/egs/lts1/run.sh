#!/bin/bash

# Copyright 2020 Academia Sinica (Pin-Jui Ku, Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1
stop_stage=100
gpu=0
ngpu=1       # number of gpu in training
nj=32        # number of parallel jobs
dumpdir=dump # directory to dump full features
verbose=1    # verbose option (if set > 1, get more log)
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs=16000      # sampling frequency
fmax=7600     # maximum frequency
fmin=80       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length

# face feature extraction related
fps=50
lip_width=128
lip_height=64
shape_predictor_path=downloads/resources/shape_predictor_68_face_landmarks.dat

# config files
train_config=conf/train_pytorch_transformer_face.yaml
decode_config=conf/decode.yaml

# normalization related
norm_name=cmvn
cmvn=../../dysarthric/vae2/exp/train_tmsv_pytorch_train_pytorch_vqvae_b/cmvn.ark

# decoding related
model=model.loss.best
n_average=0 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim
voc=PWG                     # vocoder used (GL or PWG)

# pretrained model related
vae_config=conf/train_pytorch_vqvae_b.yaml
vae_path=../../dysarthric/vae2/exp/train_tmsv_pytorch_train_pytorch_vqvae_b
vae_model=model.loss.best

# dataset related
#db_root=downloads
db_root=/mnt/md0/dataset/TMSV/raw

# objective evaluation related
outdir=
mcep_dim=24
shift_ms=5

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
dev_set=dev
eval_set=test

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data and Pretrained Model Download"

    echo "Downloading data..."
    # local/data_download.sh ${db_root}

    echo "Downloading essential resources..."
    local/resources_download.sh downloads # ${db_root}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    local/data_prep.sh ${db_root}/TMSV data/all
fi

if [ -z ${norm_name} ]; then
    echo "Please specify --norm_name ."
    exit 1
fi
feat_tr_dir=${dumpdir}/${train_set}_fbank_${norm_name}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${dev_set}_fbank_${norm_name}; mkdir -p ${feat_dt_dir}
feat_ev_dir=${dumpdir}/${eval_set}_fbank_${norm_name}; mkdir -p ${feat_ev_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"

    # make train, dev and eval sets using the lists generated in data_prep.sh
    utils/subset_data_dir.sh --utt-list data/all/train_utt_list data/all data/${train_set}
    utils/fix_data_dir.sh data/${train_set}
    utils/subset_data_dir.sh --utt-list data/all/dev_utt_list data/all data/${dev_set}
    utils/fix_data_dir.sh data/${dev_set}
    utils/subset_data_dir.sh --utt-list data/all/eval_utt_list data/all data/${eval_set}
    utils/fix_data_dir.sh data/${eval_set}
    
    # the utils/subset_data_dir.sh do not split the video.scp file for us, so we need to do this seperately
    utils/filter_scp.pl data/${train_set}/utt2spk < data/all/video.scp > data/${train_set}/video.scp
    utils/filter_scp.pl data/${dev_set}/utt2spk < data/all/video.scp > data/${dev_set}/video.scp
    utils/filter_scp.pl data/${eval_set}/utt2spk < data/all/video.scp > data/${eval_set}/video.scp

    fbankdir=fbank
    for x in $train_set $dev_set $eval_set; do
        make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            data/${x} \
            exp/make_fbank/${x} \
            ${fbankdir}
    done

    # compute statistics for global mean-variance normalization
    # If not using pretrained models statistics, calculate in a speaker-dependent way.
    if [ -z "${cmvn}" ]; then
        compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/fbank_cmvn.ark
        fbank_cmvn=data/${train_set}/fbank_cmvn.ark
    else
        fbank_cmvn=${cmvn}
    fi

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/feats.scp ${fbank_cmvn} exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${dev_set}/feats.scp ${fbank_cmvn} exp/dump_feats/dev ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}/feats.scp ${fbank_cmvn} exp/dump_feats/eval ${feat_ev_dir}
fi

face_feat_tr_dir=${dumpdir}/${train_set}_face; mkdir -p ${face_feat_tr_dir}
face_feat_dt_dir=${dumpdir}/${dev_set}_face; mkdir -p ${face_feat_dt_dir}
face_feat_ev_dir=${dumpdir}/${eval_set}_face; mkdir -p ${face_feat_ev_dir}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Face Feature Generation"

    face_feature_dir=face_feature
    for x in ${dev_set} ${eval_set} ${train_set}; do
        local/make_face.sh --cmd "${train_cmd}" --nj 32 \
            --fps ${fps} \
            --lip_width ${lip_width} \
            --lip_height ${lip_height} \
            --shape_predictor_path ${shape_predictor_path} \
            --interpolate-landmark true \
            data/${x} \
            exp/make_face/${x} \
            ${face_feature_dir}
    done
            
    # compute statistics for global mean-variance normalization for face features
    compute-cmvn-stats scp:data/${train_set}/face_feats.scp data/${train_set}/face_cmvn.ark
    face_cmvn=data/${train_set}/face_cmvn.ark

    # dump features
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/face_feats.scp ${face_cmvn} exp/dump_face_feats/${train_set} ${face_feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${dev_set}/face_feats.scp ${face_cmvn} exp/dump_face_feats/${dev_set} ${face_feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}/face_feats.scp ${face_cmvn} exp/dump_face_feats/${eval_set} ${face_feat_ev_dir}
fi

vq_feat_tr_dir=${dumpdir}/${train_set}_vqtoken; mkdir -p ${vq_feat_tr_dir}
vq_feat_dt_dir=${dumpdir}/${dev_set}_vqtoken; mkdir -p ${vq_feat_dt_dir}
vq_feat_ev_dir=${dumpdir}/${eval_set}_vqtoken; mkdir -p ${vq_feat_ev_dir}
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Extract VQVAE latent tokens form fbanks"
    for x in ${dev_set} ${eval_set} ${train_set}; do
        $VAE_BIN/extract_bnf.py \
            --config ${vae_config} \
            --model_path ${vae_path}/result/${vae_model} \
            --bnf_kind "id" \
            --output_txt "false" \
            --gpu ${gpu} \
            "scp:dump/${x}_fbank_${norm_name}/feats.scp" \
            "ark,scp:dump/${x}_vqtoken/feats.ark,dump/${x}_vqtoken/feats.scp"
    done
fi

dict_dir=data/nlsymbols
pair_tr_dir=dump/${train_set}_${norm_name}; mkdir -p ${pair_tr_dir}
pair_dt_dir=dump/${dev_set}_${norm_name}; mkdir -p ${pair_dt_dir}
pair_ev_dir=dump/${eval_set}_${norm_name}; mkdir -p ${pair_ev_dir}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Dictionary and Json Data Preparation"

    local/feats2json.sh --feat-in ${face_feat_tr_dir}/feats.scp \
        --feat-out ${vq_feat_tr_dir}/feats.scp \
        data/${train_set} > ${pair_tr_dir}/data.json
    local/feats2json.sh --feat-in ${face_feat_dt_dir}/feats.scp \
        --feat-out ${vq_feat_dt_dir}/feats.scp \
        data/${dev_set} > ${pair_dt_dir}/data.json
    local/feats2json.sh --feat-in ${face_feat_ev_dir}/feats.scp \
        --feat-out ${vq_feat_ev_dir}/feats.scp \
        data/${eval_set} > ${pair_ev_dir}/data.json
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: x-vector extraction"
    # Make MFCCs and compute the energy-based VAD for each dataset
    mfccdir=mfcc
    vaddir=mfcc
    for name in ${train_set} ${dev_set} ${eval_set}; do
        utils/copy_data_dir.sh data/${name} data/${name}_mfcc_16k
        utils/data/resample_data_dir.sh 16000 data/${name}_mfcc_16k
        steps/make_mfcc.sh \
            --write-utt2num-frames true \
            --mfcc-config conf/mfcc.conf \
            --nj ${nj} --cmd "$train_cmd" \
            data/${name}_mfcc_16k exp/make_mfcc_16k ${mfccdir}
        utils/fix_data_dir.sh data/${name}_mfcc_16k
        sid/compute_vad_decision.sh --nj ${nj} --cmd "$train_cmd" \
            data/${name}_mfcc_16k exp/make_vad ${vaddir}
        utils/fix_data_dir.sh data/${name}_mfcc_16k
    done

    # Check pretrained model existence
    nnet_dir=exp/xvector_nnet_1a
    if [ ! -e ${nnet_dir} ]; then
        echo "X-vector model does not exist. Download pre-trained model."
        wget http://kaldi-asr.org/models/8/0008_sitw_v2_1a.tar.gz
        tar xvf 0008_sitw_v2_1a.tar.gz
        mv 0008_sitw_v2_1a/exp/xvector_nnet_1a exp
        rm -rf 0008_sitw_v2_1a.tar.gz 0008_sitw_v2_1a
    fi
    # Extract x-vector
    for name in ${train_set} ${dev_set} ${eval_set}; do
        sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj ${nj} \
            ${nnet_dir} data/${name}_mfcc_16k \
            ${nnet_dir}/xvectors_${name}
    done
    # Update json
    for name in ${train_set} ${dev_set} ${eval_set}; do
        # local/update_json.sh ${dumpdir}/${name}_${norm_name}/data.json ${nnet_dir}/xvectors_${name}/xvector.scp
        echo "pass"
    done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: LTS model training"

    if [[ -z ${train_config} ]]; then
        echo "Please specify --train_config."
        exit 1
    fi

    if [ -z ${tag} ]; then
        expname=${train_set}_${backend}_$(basename ${train_config%.*})
    else
        expname=${train_set}_${backend}_${tag}
    fi
    expdir=exp/${expname}
    mkdir -p ${expdir}

    tr_json=${pair_tr_dir}/data.json
    dt_json=${pair_dt_dir}/data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        vc_train.py \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --outdir ${expdir}/results \
           --tensorboard-dir tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config} \
           --preprocess-conf conf/face_aug.yaml
fi

if [ -z "${model}" ]; then
    model="$(find "${expdir}" -name "snapshot*" -print0 | xargs -0 ls -t 2>/dev/null | head -n 1)"
else
    model=$(basename ${model})
fi
if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Decoding"

    echo "VQ token decoding..."
    pids=() # initialize pids
    for name in ${dev_set} ${eval_set}; do
    (
        [ ! -e ${outdir}/${name}_vqtoken ] && mkdir -p ${outdir}/${name}_vqtoken
        cp ${dumpdir}/${name}_${norm_name}/data.json ${outdir}/${name}_vqtoken
        splitjson.py --parts ${nj} ${outdir}/${name}_vqtoken/data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}_vqtoken/log/decode.JOB.log \
            vc_decode.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${name}_vqtoken/feats.JOB \
                --json ${outdir}/${name}_vqtoken/split${nj}utt/data.JOB.json \
                --model ${expdir}/results/${model} \
                --config ${decode_config}
        # concatenate scp files
        for n in $(seq ${nj}); do
            cat "${outdir}/${name}_vqtoken/feats.$n.scp" || exit 1;
        done > ${outdir}/${name}_vqtoken/feats.scp
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

    echo "VQVAE decoding..."
    pids=() # initialize pids
    for name in ${dev_set} ${eval_set}; do
    (
        [ ! -e ${outdir}/${name}_fbank ] && mkdir -p ${outdir}/${name}_fbank
        # decode in parallel
        $VAE_BIN/decode_bnf.py \
            --config ${vae_config} \
            --model_path ${vae_path}/result/${vae_model} \
            --bnf_kind "id" \
            --input_txt "false" \
            --utt2spk data/${name}/utt2spk \
            --spk2spk_id ${vae_path}/spk2spk_id \
            --gpu ${gpu} \
            "scp:${outdir}/${name}_vqtoken/feats.scp" \
            "ark,scp:${outdir}/${name}_fbank/feats.ark,${outdir}/${name}_fbank/feats.scp"
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

    echo "Synthesis"
    pids=() # initialize pids
    for name in ${dev_set} ${eval_set}; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}
        
        # Normalization
        # If not using pretrained models statistics, use statistics of target speaker
        if [ -n "${cmvn}" ]; then
            fbank_cmvn=${cmvn}
        else
            fbank_cmvn=data/${train_set}/fbank_cmvn.ark
        fi
        apply-cmvn --norm-vars=true --reverse=true ${fbank_cmvn} \
            scp:${outdir}/${name}_fbank/feats.scp \
            ark,scp:${outdir}_denorm/${name}/feats.ark,${outdir}_denorm/${name}/feats.scp

        # GL
        if [ ${voc} = "GL" ]; then
            echo "Using Griffin-Lim phase recovery."
            convert_fbank.sh --nj ${nj} --cmd "${train_cmd}" \
                --fs ${fs} \
                --fmax "${fmax}" \
                --fmin "${fmin}" \
                --n_fft ${n_fft} \
                --n_shift ${n_shift} \
                --win_length "${win_length}" \
                --n_mels ${n_mels} \
                --iters ${griffin_lim_iters} \
                ${outdir}_denorm/${name} \
                ${outdir}_denorm/${name}/log \
                ${outdir}_denorm/${name}/wav
        # PWG
        elif [ ${voc} = "PWG" ]; then
            echo "Using Parallel WaveGAN vocoder."

            # check existence
            # voc_expdir=${db_root}/resources/pwg
            voc_expdir=exp/parallel_wavegan
            if [ ! -d ${voc_expdir} ]; then
                echo "${voc_expdir} does not exist. Please download the pretrained model."
                exit 1
            fi

            # variable settings
            voc_checkpoint="$(find "${voc_expdir}" -name "*.pkl" -print0 | xargs -0 ls -t 2>/dev/null | head -n 1)"
            voc_conf="$(find "${voc_expdir}" -name "config.yml" -print0 | xargs -0 ls -t | head -n 1)"
            voc_stats="$(find "${voc_expdir}" -name "stats.h5" -print0 | xargs -0 ls -t | head -n 1)"
            wav_dir=${outdir}_denorm/${name}/pwg_wav
            hdf5_norm_dir=${outdir}_denorm/${name}/hdf5_norm
            [ ! -e "${wav_dir}" ] && mkdir -p ${wav_dir}
            [ ! -e ${hdf5_norm_dir} ] && mkdir -p ${hdf5_norm_dir}

            # normalize and dump them
            echo "Normalizing..."
            ${train_cmd} "${hdf5_norm_dir}/normalize.log" \
                parallel-wavegan-normalize \
                    --skip-wav-copy \
                    --config "${voc_conf}" \
                    --stats "${voc_stats}" \
                    --feats-scp "${outdir}_denorm/${name}/feats.scp" \
                    --dumpdir ${hdf5_norm_dir} \
                    --verbose "${verbose}"
            echo "successfully finished normalization."

            # decoding
            echo "Decoding start. See the progress via ${wav_dir}/decode.log."
            ${cuda_cmd} --gpu 1 "${wav_dir}/decode.log" \
                parallel-wavegan-decode \
                    --dumpdir ${hdf5_norm_dir} \
                    --checkpoint "${voc_checkpoint}" \
                    --outdir ${wav_dir} \
                    --verbose "${verbose}"
            
            # renaming
            rename -f "s/_gen//g" ${wav_dir}/*.wav

            echo "successfully finished decoding."
        else
            echo "Vocoder type not supported. Only GL and PWG are available."
        fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "stage 9: Objective Evaluation: MCD"

    for name in ${dev_set} ${eval_set}; do
        echo "Calculating MCD for ${name} set"
        num_of_spks=18
        all_mcds_file=${outdir}_denorm/${name}/all_mcd.log; rm -f ${all_mcds_file}

        for count_of_spk in $(seq 1 1 $num_of_spks); do
            spk=SP$(printf "%02d" $count_of_spk)

            out_wavdir=${outdir}_denorm/${name}/pwg_wav
            gt_wavdir=${db_root}/TMSV/${spk}/audio
            minf0=$(grep ${spk} conf/all.f0 | cut -f 2 -d" ")
            maxf0=$(grep ${spk} conf/all.f0 | cut -f 3 -d" ")
            out_spk_wavdir=${outdir}_denorm/mcd/${name}/pwg_out/${spk}
            gt_spk_wavdir=${outdir}_denorm/mcd/${name}/gt/${spk}
            mkdir -p ${out_spk_wavdir}
            mkdir -p ${gt_spk_wavdir}
           
            # copy wav files for mcd calculation
            for out_wav_file in $(find -L ${out_wavdir} -iname "${spk}_*" | sort ); do
                wav_basename=$(basename $out_wav_file .wav)
                cp ${out_wav_file} ${out_spk_wavdir} || exit 1
                cp ${gt_wavdir}/${wav_basename}.wav ${gt_spk_wavdir}
            done

            # actual calculation
            mcd_file=${outdir}_denorm/${name}/${spk}_mcd.log
            ${decode_cmd} ${mcd_file} \
                mcd_calculate.py \
                    --wavdir ${out_spk_wavdir} \
                    --gtwavdir ${gt_spk_wavdir} \
                    --mcep_dim ${mcep_dim} \
                    --shiftms ${shift_ms} \
                    --f0min ${minf0} \
                    --f0max ${maxf0}
            grep "Mean MCD" < ${mcd_file} >> ${all_mcds_file}
            echo "${spk}: $(grep 'Mean MCD' < ${mcd_file})"
        done
        echo "Mean MCD for ${name} set is $(awk '{ total += $3; count++ } END { print total/count }' ${all_mcds_file})"
    done
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "Stage 10: objective evaluation: ASR"
    
    for name in ${dev_set} ${eval_set}; do
        local/ob_eval/evaluate.sh \
            --db_root ${db_root} \
            --vocoder ${voc} \
            ${outdir} ${name}
    done
fi

