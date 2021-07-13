#!/bin/bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
# [stage 6] 2019 Okayama University (Katsuki Inoue)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0
stop_stage=2
ngpu=1       # number of gpus ("0" uses cpu, otherwise use gpu)
nj=32        # number of parallel jobs
dumpdir=dump # directory to dump full features
verbose=1    # verbose option (if set > 0, get more log)
N=0          # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)
debugmode=1

# feature extraction related
fs=22050      # sampling frequency
fmax="" #7600     # maximum frequency
fmin="" #80       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length

# char or phn
# In the case of phn, input transcription is convered to phoneem using https://github.com/Kyubyong/g2p.
trans_type="char"

# config files
train_config=conf/tuning/train_pytorch_transformer_rev2.yaml # you can select from conf or conf/tuning.
# train_pytorch_transformer train_rnn_rev3
                                               # now we support tacotron2, transformer, and fastspeech
                                               # see more info in the header of each config.
decode_config=conf/tuning/decode_pytorch_transformer.yaml
# decode_pytorch_transformer decode_rnn

# decoding related
model=model.loss.best
# n_average=1 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
# griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# # objective evaluation related
# asr_model="librispeech.transformer.ngpu4"
# eval_tts_model=true                            # true: evaluate tts model, false: evaluate ground truth
# wer=true                                       # true: evaluate CER & WER, false: evaluate only CER

# root directory of db
db_root=/abelab/DB4 #downloads

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="${trans_type}_train_no_dev"
dev_set="${trans_type}_dev"
eval_set="${trans_type}_eval"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    local/data_download.sh ${db_root}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/data_prep.sh ${db_root}/LJSpeech-1.1 data/${trans_type}_train ${trans_type}
    utils/validate_data_dir.sh --no-feats data/${trans_type}_train
fi

feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${dev_set}; mkdir -p ${feat_dt_dir}
feat_ev_dir=${dumpdir}/${eval_set}; mkdir -p ${feat_ev_dir}
cmvn="data/${train_set}/cmvn.ark"
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev name by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"

    # Generate the fbank features; by default 80-dimensional fbanks on each frame
    fbankdir=fbank
    make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
        --fs ${fs} \
        --fmax "${fmax}" \
        --fmin "${fmin}" \
        --n_fft ${n_fft} \
        --n_shift ${n_shift} \
        --win_length "${win_length}" \
        --n_mels ${n_mels} \
        data/${trans_type}_train \
        exp/${trans_type}_make_fbank/train \
        ${fbankdir}

    # make a dev set
    utils/subset_data_dir.sh --last data/${trans_type}_train 500 data/${trans_type}_deveval
    utils/subset_data_dir.sh --last data/${trans_type}_deveval 250 data/${eval_set}
    utils/subset_data_dir.sh --first data/${trans_type}_deveval 250 data/${dev_set}
    n=$(( $(wc -l < data/${trans_type}_train/wav.scp) - 500 ))
    utils/subset_data_dir.sh --first data/${trans_type}_train ${n} data/${train_set}

    # compute statistics for global mean-variance normalization
    compute-cmvn-stats scp:data/${train_set}/feats.scp ${cmvn}

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${train_set}/feats.scp ${cmvn} exp/dump_feats/${trans_type}_train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${dev_set}/feats.scp ${cmvn} exp/dump_feats/${trans_type}_dev ${feat_dt_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}/feats.scp ${cmvn} exp/dump_feats/${trans_type}_eval ${feat_ev_dir}
fi

dict=data/lang_1${trans_type}/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1${trans_type}/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 --trans_type ${trans_type} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --trans_type ${trans_type} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --trans_type ${trans_type} \
         data/${dev_set} ${dict} > ${feat_dt_dir}/data.json
    data2json.sh --feat ${feat_ev_dir}/feats.scp --trans_type ${trans_type} \
         data/${eval_set} ${dict} > ${feat_ev_dir}/data.json
fi


if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

# if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#     echo "stage 3: LM Preparation"
#     lmdatadir=data/local/lm_train_${bpemode}${nbpe}
#     [ ! -e ${lmdatadir} ] && mkdir -p ${lmdatadir}
#     gunzip -c db/TEDLIUM_release2/LM/*.en.gz | sed 's/ <\/s>//g' | local/join_suffix.py |\
# 	spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
#     cut -f 2- -d" " data/${train_dev}/text | spm_encode --model=${bpemodel}.model --output_format=piece \
# 	> ${lmdatadir}/valid.txt
#     ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
#         lm_train.py \
#         --config ${lm_config} \
#         --ngpu ${ngpu} \
#         --backend ${backend} \
#         --verbose 1 \
#         --outdir ${lmexpdir} \
#         --tensorboard-dir tensorboard/${lmexpname} \
#         --train-label ${lmdatadir}/train.txt \
#         --valid-label ${lmdatadir}/valid.txt \
#         --resume ${lm_resume} \
#         --dict ${dict}
# fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    tr_json=${feat_tr_dir}/data.json # ${feat_tr_dir}/data_${bpemode}${nbpe}.json
    dt_json=${feat_dt_dir}/data.json # ${feat_tr_dir}/data_${bpemode}${nbpe}.json

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --backend ${backend} \
        --ngpu ${ngpu} \
        --minibatches ${N} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --verbose ${verbose} \
        --debugmode ${debugmode} \
        --debugdir ${expdir} \
        --resume ${resume} \
        --train-json ${tr_json} \
        --valid-json ${dt_json} \
        --dict ${dict} \
        --config ${train_config} \
        --ctc_type builtin 

fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=8

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*}) #_${lmtag}
        # if [ ${use_wordlm} = true ]; then
        #     recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        # else
        #     recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        # fi
        feat_recog_dir=${dumpdir}/${rtask}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --verbose ${verbose} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} 
            
            # ${recog_opts}

        score_sclite.sh ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi
