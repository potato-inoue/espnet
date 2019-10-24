#!/bin/bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Begin configuration section.
nj=4
fs=22050
n_fft=1024
n_shift=512
win_length=
n_mels=
cmd=run.pl
help_message=$(cat <<EOF
Usage: $0 [options] <data-dir> [<log-dir> [<fbank-dir>] ]
e.g.: $0 data/train exp/griffin_lim/train wav
Note: <log-dir> defaults to <data-dir>/log, and <fbank-dir> defaults to <data-dir>/data
Options:
  --nj <nj>                  # number of parallel jobs
  --fs <fs>                  # sampling rate
  --fmax <fmax>              # maximum frequency
  --fmin <fmin>              # minimum frequency
  --n_fft <n_fft>            # number of FFT points (default=1024)
  --n_shift <n_shift>        # shift size in point (default=256)
  --win_length <win_length>  # window length in point (default=)
  --n_mels <n_mels>          # number of mel basis (default=80)
  --iters <iters>            # number of Griffin-lim iterations (default=64)
  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs.
EOF
)
# End configuration section.

echo "$0 $*"  # Print the command line for logging

. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 4 ]; then
    echo "${help_message}"
    exit 1;
fi

data_tgt=$1
data_hyp=$2
if [ $# -ge 3 ]; then
  logdir=$3
else
  logdir=${data}/log
fi
if [ $# -ge 4 ]; then
  msedir=$4
else
  msedir=${data}/data
fi

# use "name" as part of name of the archive.
name_tgt=$(basename ${data_tgt})
name_hyp=$(basename ${data_hyp})

mkdir -p ${msedir} || exit 1;
mkdir -p ${logdir} || exit 1;

scp_tgt=${data_tgt}/feats.scp
scp_hyp=${data_hyp}/feats.scp

split_scps_tgt=""
split_scps_hyp=""
for n in $(seq ${nj}); do
    split_scps_tgt="$split_scps_tgt $logdir/feats.$n.tgt.scp"
    split_scps_hyp="$split_scps_hyp $logdir/feats.$n.hyp.scp"
done

utils/split_scp.pl ${scp_tgt} ${split_scps_tgt} || exit 1;
utils/split_scp.pl ${scp_hyp} ${split_scps_hyp} || exit 1;

${cmd} JOB=1:${nj} ${logdir}/mce_eval_${name}.JOB.log \
    objective_eval.py \
        --fs ${fs} \
        --win_length ${win_length} \
        --n_fft ${n_fft} \
        --n_shift ${n_shift} \
        --n_mels ${n_mels} \
        scp:${logdir}/feats.JOB.tgt.scp \
        scp:${logdir}/feats.JOB.hyp.scp \
        ${msedir}

# rm ${logdir}/feats.*.scp 2>/dev/null

echo "Succeeded creating wav for $name"
