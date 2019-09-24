#!/bin/bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=8
stop_stage=8
ngpu=1       # number of gpus ("0" uses cpu, otherwise use gpu)
nj=32        # numebr of parallel jobs
dumpdir=dump # directory to dump full features
verbose=0    # verbose option (if set > 0, get more log)
N=0          # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs=22050      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length

# feature configuration
do_delta=false

# config files
# train_config=conf/train_pytorch_tacotron2.yaml # you can select from conf or conf/tuning.
#                                                # now we support tacotron2, transformer, and fastspeech
#                                                # see more info in the header of each config.
# decode_config=conf/decode.yaml
tts_train_config="conf/tuning/tts_train_pytorch_transformer.fine-tuning.rev0.yaml"
tts_decode_config="conf/tts_decode.fine-tuning.yaml"
asr_decode_config="conf/asr_decode.fine-tuning.yaml"

# decoding related
model=model.loss.best
n_average=1 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# root directory of db
db_root=/export/a06/katsuki/DB #downloads

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

prep_set="dev-clean test-clean dev-other test-other"
extract_set="test-clean dev-clean"

# pre-trained model urls for downloads.
asr_model="librispeech.transformer.ngpu4"
tts_model="ljspeech.transformer.v1"

# Cut function
tts_fbank=true   # First half of stage 4 
tts_dump=true   # Last half of stage 4 

# speaker selection
set_name="test_clean_22050"
spk="237"
train_num=250
dev_num=14
eval_num=15

# auto setting 
deveval_num=$(($dev_num + $eval_num))
tts_data_set="${set_name}_${spk}"
train_set="${tts_data_set}_train_no_dev"
dev_set="${tts_data_set}_dev"
eval_set="${tts_data_set}_eval"

asr_model_dir=exp/asr/${asr_model}
case "${asr_model}" in
    # "librispeech.transformer.ngpu1") asr_url="https://drive.google.com/open?id=1bOaOEIZBveERti0x6mnBYiNsn6MSRd2E" \
    #   asr_cmvn="${asr_model_dir}/data/train_960/cmvn.ark" \
    #   asr_pre_decode_config="${asr_model_dir}/conf/tuning/decode_pytorch_transformer.yaml" \ 
    #   recog_model="${asr_model_dir}/exp/train_960_pytorch_train_pytorch_transformer_lr5.0_ag8.v2/results/model.last10.avg.best" \
    #   lang_model="${asr_model_dir}/exp/train_rnnlm_pytorch_lm_unigram5000/rnnlm.model.best" ;;
  
    "librispeech.transformer.ngpu4") asr_url="https://drive.google.com/open?id=1BtQvAnsFvVi-dp_qsaFP7n4A_5cwnlR6" \
      asr_cmvn="${asr_model_dir}/data/train_960/cmvn.ark" \
      asr_pre_decode_config="${asr_model_dir}/conf/tuning/decode_pytorch_transformer_large.yaml" \
      recog_model="${asr_model_dir}/exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug/results/model.val5.avg.best" \
      lang_model="${asr_model_dir}/exp/irielm.ep11.last5.avg/rnnlm.model.best" ;;        
    
    *) echo "No such models: ${asr_model}"; exit 1 ;;
esac
echo "ASR model: ${asr_model_dir}"

tts_model_dir=exp/tts/${tts_model}
case "${tts_model}" in
    "ljspeech.transformer.v1") tts_url="https://drive.google.com/open?id=13DR-RB5wrbMqBGx_MC655VZlsEq52DyS" 
      tts_dict="${tts_model_dir}/data/lang_1char/train_no_dev_units.txt" \
      tts_cmvn="${tts_model_dir}/data/train_no_dev/cmvn.ark" \
      tts_pre_train_config="${tts_model_dir}/conf/tuning/train_pytorch_transformer.v1.yaml" \
      tts_pre_decode_config="${tts_model_dir}/conf/decode.yaml" \
      tts_model_conf="${tts_model_dir}/exp/train_no_dev_pytorch_train_pytorch_transformer.v1/results/model.json"
      synth_model="${tts_model_dir}/exp/train_no_dev_pytorch_train_pytorch_transformer.v1/results/model.last1.avg.best" ;;       

    *) echo "No such models: ${tts_model}"; exit 1 ;;
esac
echo "TTS model: ${tts_model_dir}"

if [ ${stage} -le -2 ] && [ ${stop_stage} -ge -2 ]; then
    echo "stage -2: Pre-trained model Download"
  
    if [ ! -e ${asr_model_dir}/.complete ]; then
        mkdir -p ${asr_model_dir}
        download_from_google_drive.sh ${asr_url} ${asr_model_dir} ".tar.gz"
        touch ${asr_model_dir}/.complete
    fi
  
    if [ ! -e ${tts_model_dir}/.complete ]; then
        mkdir -p ${tts_model_dir}
        download_from_google_drive.sh ${tts_url} ${tts_model_dir} ".tar.gz"
        touch ${tts_model_dir}/.complete
    fi
fi

# if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
#     echo "stage -1: Data Download"
#     local/download.sh ${db_root}
# fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: ASR&TTS Data Preparation"

    for part in ${prep_set}; do
        # use underscore-separated names in data directories.
        # local/data_prep_asr.sh ${db_root}/LibriTTS/${part} data/asr/${part//-/_}
        local/data_prep_tts.sh ${db_root}/LibriTTS/${part} data/tts/${part//-/_}
        mv data/tts/${part//-/_} data/tts/${part//-/_}_org
        
        utils/copy_data_dir.sh data/tts/${part//-/_}_org data/tts/${part//-/_}_org_22050
        utils/data/resample_data_dir.sh ${fs} data/tts/${part//-/_}_org_22050
    done
fi

echo "stage 1:(ASR) Feature Generation"
echo "stage 2:(ASR) Dictionary and Json Data Preparation"
echo "stage 3:(ASR) Decoding"

tts_feat_tr_dir=${dumpdir}/tts/${train_set}; mkdir -p ${tts_feat_tr_dir}
tts_feat_dt_dir=${dumpdir}/tts/${dev_set}; mkdir -p ${tts_feat_dt_dir}
tts_feat_ev_dir=${dumpdir}/tts/${eval_set}; mkdir -p ${tts_feat_ev_dir}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    ### Task dependent. You have to design training and dev name by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 4:(TTS) Feature Generation"

    if [ ${tts_fbank} == 'true' ]; then
        # Generate the fbank features; by default 80-dimensional fbanks on each frame
        fbankdir=fbank/tts
        # for part in ${prep_set}; do
        #     make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
        #         --fs ${fs} \
        #         --fmax "${fmax}" \
        #         --fmin "${fmin}" \
        #         --n_fft ${n_fft} \
        #         --n_shift ${n_shift} \
        #         --win_length "${win_length}" \
        #         --n_mels ${n_mels} \
        #         data/tts/${part//-/_}_org_22050 \
        #         exp/tts/make_fbank/${part//-/_}_org_22050 \
        #         ${fbankdir}
        # done

        for x in ${extract_set//-/_}; do
            asr_result_dir=exp/asr/decode_result/${x}

            #Replace txt to recog_txt
            concatjson.py ${asr_result_dir}/result.*.json > ${asr_result_dir}/result.json
            local/make_tts_text.sh ${asr_result_dir} data/tts/${x}_org_22050

            #remove utt having more than 3000 frames
            #remove utt having more than 400 characters
            #remove utt having less than 1 charactors (= no recognized text)
            remove_longshortdata.sh --maxframes 3000 --maxchars 400 --minchars 1 data/tts/${x}_org_22050 data/tts/${x}_22050

            if [ -e exp/tts/spk_list/super ]; then
                rm exp/tts/spk_list/super/*
            else
                mkdir -p exp/tts/spk_list/super
            fi
            if [ -e exp/tts/spk_list/sub ]; then
                rm exp/tts/spk_list/sub/*
            else
                mkdir -p exp/tts/spk_list/sub
            fi

            cat data/tts/${x}_22050/spk2utt | awk '{print $1}' > exp/tts/spk_list/super/${x}_22050.txt
            cat exp/tts/spk_list/super/${x}_22050.txt | while read -r line; do
                spk=$(echo $line | awk -F'_' '{print $1}')
                echo $line >> exp/tts/spk_list/sub/${x}_22050_${spk}.txt
            done
            find exp/tts/spk_list/sub/ -name "${x}_22050_*.txt" | sort | while read -r line; do
                spk=$(head -n 1 $line | awk -F'_' '{print $1}')
                utils/subset_data_dir.sh --spk-list ${line} data/tts/${x}_22050 data/tts/${x}_22050_${spk}
            done
        done

    fi

    if [ ${tts_dump} == 'true' ]; then
        # make a dev set
        utils/subset_data_dir.sh --last data/tts/${tts_data_set} ${deveval_num} data/tts/${tts_data_set}_deveval
        utils/subset_data_dir.sh --last data/tts/${tts_data_set}_deveval ${eval_num} data/tts/${eval_set}
        utils/subset_data_dir.sh --first data/tts/${tts_data_set}_deveval ${dev_num} data/tts/${dev_set}
        n=$(( $(wc -l < data/tts/${tts_data_set}/wav.scp) - ${deveval_num} ))
        utils/subset_data_dir.sh --first data/tts/${tts_data_set} ${n} data/tts/${train_set}

        # compute statistics for global mean-variance normalization
        # compute-cmvn-stats scp:data/tts/${train_set}/feats.scp data/tts/${train_set}/cmvn.ark

        # dump features for training
        dump.sh --cmd "$train_cmd" --nj 1 --do_delta false \
            data/tts/${train_set}/feats.scp ${tts_cmvn} exp/tts/dump_feats/${tts_data_set}_train ${tts_feat_tr_dir}
        dump.sh --cmd "$train_cmd" --nj 1 --do_delta false \
            data/tts/${dev_set}/feats.scp ${tts_cmvn} exp/tts/dump_feats/${tts_data_set}_dev ${tts_feat_dt_dir}
        dump.sh --cmd "$train_cmd" --nj 1 --do_delta false \
            data/tts/${eval_set}/feats.scp ${tts_cmvn} exp/tts/dump_feats/${tts_data_set}_eval ${tts_feat_ev_dir}
    fi
fi

echo "dictionary: ${tts_dict}"
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 5:(TTS) Dictionary and Json Data Preparation"

    mkdir -p data/tts/lang_1char/
    cp ${tts_dict} data/tts/lang_1char/${tts_dict##*/}
    wc -l ${tts_dict}

    # make json labels
    data2json.sh --feat ${tts_feat_tr_dir}/feats.scp \
        data/tts/${train_set} ${tts_dict} > ${tts_feat_tr_dir}/data.json
    data2json.sh --feat ${tts_feat_dt_dir}/feats.scp \
        data/tts/${dev_set} ${tts_dict} > ${tts_feat_dt_dir}/data.json
    data2json.sh --feat ${tts_feat_ev_dir}/feats.scp \
        data/tts/${eval_set} ${tts_dict} > ${tts_feat_ev_dir}/data.json
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${tts_train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/tts/${expname}; mkdir -p ${expdir}
outdir=${expdir}/outputs_${tts_model}_$(basename ${tts_decode_config%.*})_0th
[ ! -e ${expdir}/results ] && mkdir -p ${expdir}/results
[ ! -e ${expdir}/results/model.0th.copy ] && cp ${synth_model} ${expdir}/results/model.0th.copy
[ ! -e ${tts_decode_config} ] && cp ${tts_pre_decode_config} ${tts_decode_config}
[ ! -e ${expdir}/results/model.json ] && cp ${tts_model_conf} ${expdir}/results/model.json
pre_trained_tts_model=${expdir}/results/model.0th.copy
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6:(TTS) 0th Decoding"

    pids=() # initialize pids
    for name in ${dev_set} ${eval_set}; do #${dev_set} ${eval_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${dumpdir}/tts/${name}/data.json ${outdir}/${name}
        splitjson.py --parts ${nj} ${outdir}/${name}/data.json
        
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            tts_decode.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${name}/feats.JOB \
                --json ${outdir}/${name}/split${nj}utt/data.JOB.json \
                --model ${expdir}/results/model.0th.copy \
                --config ${tts_decode_config}
        
        # concatenate scp files
        for n in $(seq ${nj}); do
            cat "${outdir}/${name}/feats.$n.scp" || exit 1;
        done > ${outdir}/${name}/feats.scp
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7:(TTS) 0th Synthesis"

    pids=() # initialize pids
    for name in ${dev_set} ${eval_set}; do #${dev_set} ${eval_set}; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}
        apply-cmvn --norm-vars=true --reverse=true ${tts_cmvn} \
            scp:${outdir}/${name}/feats.scp \
            ark,scp:${outdir}_denorm/${name}/feats.ark,${outdir}_denorm/${name}/feats.scp
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
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi

if [ -z ${pre_trained_tts_model} ]; then
  echo "Start TTS fine-tuning from ${pre_trained_tts_model}"
else   
  echo "Resume TTS training from ${pre_trained_tts_model}"
fi
if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${tts_train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/tts/${expname}; mkdir -p ${expdir}
outdir=${expdir}/outputs_${tts_model}_$(basename ${tts_decode_config%.*})
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8:(TTS) Model Training"

    cat ${tts_pre_train_config} | sed -e "s/epochs: 100/epochs: 10/" \
    | sed -e "s/# other/# other\nreport-interval-iters: 1/" \
    | sed -e "s/save-interval-epoch: 10/save-interval-epoch: 1/" > ${tts_train_config}
    # | sed -e 's/transformer-lr: 1.0/transformer-lr: 1e-8/' 
    

    tr_json=${tts_feat_tr_dir}/data.json
    dt_json=${tts_feat_dt_dir}/data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        tts_train.py \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --minibatches ${N} \
           --outdir ${expdir}/results \
           --tensorboard-dir tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --model ${pre_trained_tts_model} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${tts_train_config}
fi

if [ ${n_average} -gt 0 ]; then
    model=model.last${n_average}.avg.best
fi
outdir=${expdir}/outputs_${model}_$(basename ${tts_decode_config%.*})
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "stage 9:(TTS) Decoding"
    if [ ${n_average} -gt 0 ]; then
        average_checkpoints.py --backend ${backend} \
                               --snapshots ${expdir}/results/snapshot.ep.* \
                               --out ${expdir}/results/${model} \
                               --num ${n_average}
    fi
    pids=() # initialize pids
    for name in ${dev_set} ${eval_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${dumpdir}/${name}/data.json ${outdir}/${name}
        splitjson.py --parts ${nj} ${outdir}/${name}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            tts_decode.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${name}/feats.JOB \
                --json ${outdir}/${name}/split${nj}utt/data.JOB.json \
                --model ${expdir}/results/${model} \
                --config ${decode_config}
        # concatenate scp files
        for n in $(seq ${nj}); do
            cat "${outdir}/${name}/feats.$n.scp" || exit 1;
        done > ${outdir}/${name}/feats.scp
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "stage 10:(TTS) Synthesis"
    pids=() # initialize pids
    for name in ${dev_set} ${eval_set}; do
    (
        [ ! -e ${outdir}_denorm/${name} ] && mkdir -p ${outdir}_denorm/${name}
        apply-cmvn --norm-vars=true --reverse=true data/tts/${train_set}/cmvn.ark \
            scp:${outdir}/${name}/feats.scp \
            ark,scp:${outdir}_denorm/${name}/feats.ark,${outdir}_denorm/${name}/feats.scp
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
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished."
fi
