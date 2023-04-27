#!/bin/bash

# Copyright 2021 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Begin configuration section.
nj=4
fps=
target_utt2num_frames=""
interp_mode=bilinear
lip_width=
lip_height=
shape_predictor_path=
write_utt2num_frames=true
cmd=run.pl
compress=true
filetype=mat # mat or hdf5
# End configuration section.

help_message=$(cat <<EOF
Usage: $0 [options] <data-dir> [<log-dir> [<face_feature-dir>] ]
e.g.: $0 data/train exp/make_face/train face_feature
Note: <log-dir> defaults to <data-dir>/log, and <fbank-dir> defaults to <data-dir>/data
Options:
  --nj <nj>                                        # number of parallel jobs
  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs.
  --filetype <mat|hdf5|sound.hdf5>                 # Specify the format of feats file
EOF
)
echo "$0 $*"  # Print the command line for logging

. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "${help_message}"
    exit 1;
fi

set -euo pipefail

data=$1
if [ $# -ge 2 ]; then
  logdir=$2
else
  logdir=${data}/log
fi
if [ $# -ge 3 ]; then
  facedir=$3
else
  facedir=${data}/data
fi

# make $facedir an absolute pathname.
facedir=$(perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' ${facedir} ${PWD})

# use "name" as part of name of the archive.
name=$(basename ${data})

mkdir -p ${facedir} || exit 1;
mkdir -p ${logdir} || exit 1;

if [ -f ${data}/face_feats.scp ]; then
  mkdir -p ${data}/.backup
  echo "$0: moving $data/feats.scp to $data/.backup"
  [ -f ${data}/face_feats.scp ] && mv ${data}/face_feats.scp ${data}/.backup
fi

scp=${data}/video.scp

utils/validate_data_dir.sh --no-wav --no-text --no-feats ${data} || exit 1;

if [ -f "target_utt2num_frames" ]; then
    target_utt2num_frames_opt="--target_utt2num_frames ${target_utt2num_frames}"
else
    target_utt2num_frames_opt=
fi

split_scps=""
for n in $(seq ${nj}); do
    split_scps="${split_scps} ${logdir}/video.${n}.scp"
done

utils/split_scp.pl ${scp} ${split_scps} || exit 1;

if ${write_utt2num_frames}; then
  write_num_frames_opt="--write-num-frames=ark,t:${logdir}/face_utt2num_frames.JOB"
else
  write_num_frames_opt=
fi

if [ "${filetype}" == hdf5 ]; then
    ext=h5
else
    ext=ark
fi


${cmd} JOB=1:${nj} ${logdir}/make_face_${name}.JOB.log \
    local/compute-face-feats.py \
        --fps ${fps} \
        ${target_utt2num_frames_opt} \
        --interp_mode ${interp_mode} \
        --lip_width ${lip_width} \
        --lip_height ${lip_height} \
        --shape_predictor_path ${shape_predictor_path} \
        ${write_num_frames_opt} \
        --compress=${compress} \
        --filetype ${filetype} \
        ${logdir}/video.JOB.scp \
        ark,scp:${facedir}/raw_face_${name}.JOB.${ext},${facedir}/raw_face_${name}.JOB.scp \
        ${facedir}/face_info_${name}.JOB

# concatenate the .scp files together.
for n in $(seq ${nj}); do
    cat ${facedir}/raw_face_${name}.${n}.scp || exit 1;
done > ${data}/face_feats.scp || exit 1

# concatenate the info files together.
for n in $(seq ${nj}); do
    cat ${facedir}/face_info_${name}.${n} || exit 1;
done > ${data}/face_info || exit 1

if ${write_utt2num_frames}; then
    for n in $(seq ${nj}); do
        cat ${logdir}/face_utt2num_frames.${n} || exit 1;
    done > ${data}/face_utt2num_frames || exit 1
    rm ${logdir}/face_utt2num_frames.* 2>/dev/null
fi

rm -f ${logdir}/video.*.scp ${logdir}/segments.* 2>/dev/null

# Write the filetype, this will be used for data2json.sh
echo ${filetype} > ${data}/filetype

nf=$(wc -l < ${data}/face_feats.scp)
nu=$(wc -l < ${data}/video.scp)
if [ ${nf} -ne ${nu} ]; then
    echo "It seems not all of the feature files were successfully ($nf, $nl, $nu are not equal);"
    echo "consider using utils/fix_data_dir.sh $data"
fi

echo "Succeeded creating face features for $name"
