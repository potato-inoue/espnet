#!/bin/bash

# Copyright 2019 Okayama University (Katsuki Inoue)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=$1
stop_stage=$2
ngpu=1        # number of gpu in training
nj=32 #32 #16     # numebr of parallel jobs
dumpdir=dump  # directory to dump full features
verbose=1     # verbose option (if set > 1, get more log)
seed=1        # random seed number
tts_resume="" # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs=24000      # sampling frequency
fmax=""       # maximum frequency
fmin=""       # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024    # number of fft points
n_shift=256   # number of shift points
win_length="" # window length

# feature configuration
do_delta=false

# speaker selection
set_name=$6
spk=$3
dev_num=$4
eval_num=$5
# tts config files
tts_train_config="conf/tuning/tts_train_pytorch_transformer.fine-tuning.spk${spk}_lr1e0.rev1.yaml"
# tts_train_config="conf/tuning/tts_train_pytorch_transformer.fine-tuning.spk${spk}_lr${1}.yaml"
# tts_train_config="conf/tuning/tts_train_pytorch_transformer.fine-tuning.rev8.yaml"
tts_decode_config="conf/tts_decode.fine-tuning.yaml"
asr_decode_config="conf/asr_decode.fine-tuning.yaml"

# decoding related
tts_model=model.loss.best
n_average=1           # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# Set this to somewhere where you want to put your data, or where
# someone else has already put it. You'll want to change this
# if you're not on the CLSP grid.
datadir=/export/a06/katsuki/DB

# base url for downloads.
data_url=

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Set selection
# dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500
all_set="dev-clean test-clean train-clean-100 train-clean-360"
download_set=""
prep_set="test-clean" #"test-clean test-other"
extract_set="test-clean" #"test-other" #"dev-clean dev-other"

# pre-trained model urls for downloads.
asr_model="librispeech.transformer.ngpu4"
tts_model="libritts.transformer.v1"

# Cut function
tts_fbank=false # First half of stage 4 
tts_dump=true   # Last half of stage 4 

# objective evaluation related
asr_eval_model="librispeech.transformer.ngpu4"
eval_tts_model=1                               # 1:evaluate tts model, 0:evaluate ground truth
voc="GL"                                       # the selection of vocoder
api="v2"                                       # v1: w/ att_ws generation, v2: w/o att_ws generation

# auto setting 
deveval_num=$(($dev_num + $eval_num))
tts_data_set="${set_name}_${spk}"
train_set="${tts_data_set}_train_no_dev"
dev_set="${7}_${spk}_dev"
# eval_set="${8}_${spk}_eval"
eval_set="${8}_${spk}_eval_x"

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
    "libritts.transformer.v1") tts_url="https://drive.google.com/open?id=1Xj73mDPuuPH8GsyNO8GnOC3mn0_OK4g3" 
      tts_dict="${tts_model_dir}/data/lang_1char/train_clean_460_units.txt" \
      tts_cmvn="${tts_model_dir}/data/train_clean_460/cmvn.ark" \
      tts_pre_train_config="${tts_model_dir}/conf/train_pytorch_transformer+spkemb.yaml" \
      tts_pre_decode_config="${tts_model_dir}/conf/decode.yaml" \
      tts_model_conf="${tts_model_dir}/exp/train_clean_460_pytorch_train_pytorch_transformer+spkemb.v5/results/model.json"
      synth_model="${tts_model_dir}/exp/train_clean_460_pytorch_train_pytorch_transformer+spkemb.v5/results/model.last1.avg.best" ;;       
    
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

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    mkdir -p ${datadir}
  
    for part in ${download_set}; do
        local/download_and_untar.sh ${datadir} ${data_url} ${part}
    done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: ASR&TTS Data Preparation"
    
    for part in ${prep_set}; do
        # use underscore-separated names in data directories.
        # local/data_prep_asr.sh ${datadir}/LibriTTS/${part} data/asr/${part//-/_}_org
        local/data_prep_tts.sh ${datadir}/LibriTTS/${part} data/tts/${part//-/_}_org
    done

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev name by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1:(ASR) Feature Generation"
    
    asr_fbankdir="fbank/asr"
    for x in ${extract_set//-/_}; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} \
          --write_utt2num_frames true \
          data/asr/${x} \
          exp/asr/make_fbank/${x} \
          ${asr_fbankdir}
        
        utils/fix_data_dir.sh data/asr/${x}

        asr_feat_dir=${dumpdir}/asr/${x}; mkdir -p ${asr_feat_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
          data/asr/${x}/feats.scp ${asr_cmvn} exp/asr/dump_feats/${x} \
          ${asr_feat_dir}
    done
    
fi


asr_dict=data/asr/decode_dict/X.txt; mkdir -p ${asr_dict%/*}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2:(ASR) Dictionary and Json Data Preparation"
    
    for x in ${extract_set//-/_}; do
        asr_feat_dir=${dumpdir}/asr/${x}

        echo "<unk> 1" > ${asr_dict}
        data2json.sh --feat ${asr_feat_dir}/feats.scp \
          data/asr/${x} ${asr_dict} > ${asr_feat_dir}/data.json
    done
    
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3:(ASR) Decoding"
    
    cat ${asr_pre_decode_config} | sed -e 's/beam-size: 60/beam-size: 10/' > ${asr_decode_config}
    
    for x in ${extract_set//-/_}; do
        asr_feat_dir=${dumpdir}/asr/${x}
        asr_result_dir=exp/asr/decode_result/${x}

        # split data
        splitjson.py --parts ${nj} ${asr_feat_dir}/data.json
    
        # set batchsize 0 to disable batch decoding    
        ${decode_cmd} JOB=1:${nj} ${asr_result_dir}/log/decode.JOB.log \
            asr_recog.py \
              --config ${asr_decode_config} \
              --ngpu 0 \
              --backend ${backend} \
              --batchsize 0 \
              --recog-json ${asr_feat_dir}/split${nj}utt/data.JOB.json \
              --result-label ${asr_result_dir}/result.JOB.json \
              --model ${recog_model} \
              --api v2 \
              --rnnlm ${lang_model}
    done

fi


tts_feat_tr_dir=${dumpdir}/tts/${train_set}; mkdir -p ${tts_feat_tr_dir}
tts_feat_dt_dir=${dumpdir}/tts/${dev_set}; mkdir -p ${tts_feat_dt_dir}
tts_feat_ev_dir=${dumpdir}/tts/${eval_set}; mkdir -p ${tts_feat_ev_dir}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4:(TTS) Feature Generation"
    nj=32
    if [ ${tts_fbank} == 'true' ]; then
        tts_fbankdir="fbank/tts"
        for x in ${extract_set//-/_}; do
            make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
              --fs ${fs} \
              --fmax "${fmax}" \
              --fmin "${fmin}" \
              --n_fft ${n_fft} \
              --n_shift ${n_shift} \
              --win_length "${win_length}" \
              --n_mels ${n_mels} \
              data/tts/${x}_org \
              exp/tts/make_fbank/${x}_org \
              ${tts_fbankdir}
        done

        for x in ${extract_set//-/_}; do
            asr_result_dir=exp/asr/decode_result/${x}

            # Adapt for ljspeech
            utils/copy_data_dir.sh data/tts/${x}_org data/tts/${x}_gt
            cp data/tts/${x}_gt/text data/tts/${x}_gt/text_org

            #Replace txt to recog_txt
            concatjson.py ${asr_result_dir}/result.*.json > ${asr_result_dir}/result.json
            utils/copy_data_dir.sh data/tts/${x}_org data/tts/${x}_asr
            local/make_tts_text.sh ${asr_result_dir} data/tts/${x}_asr
            
            #remove utt having more than 3000 frames
            #remove utt having more than 400 characters
            #remove utt having less than 1 charactors (= no recognized text)
            remove_longshortdata.sh --maxframes 3000 --maxchars 400 --minchars 1 data/tts/${x}_asr data/tts/${x}_asr_rm_max
            remove_longshortdata.sh --maxframes 3000 --maxchars 400 --minchars 1 data/tts/${x}_gt data/tts/${x}_gt_rm_max

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

            cat data/tts/${x}_org/spk2utt | awk '{print $1}' > exp/tts/spk_list/super/${x}.txt
            cat exp/tts/spk_list/super/${x}.txt | while read -r line; do
                spk=$(echo $line | awk -F'_' '{print $1}')
                echo $line >> exp/tts/spk_list/sub/${x}_${spk}.txt
            done
            find exp/tts/spk_list/sub/ -name "${x}_*.txt" | sort | while read -r line; do
                spk=$(head -n 1 $line | awk -F'_' '{print $1}')
                utils/subset_data_dir.sh --spk-list ${line} data/tts/${x}_asr_rm_max data/tts/${x}_asr_rm_max_${spk}
                utils/subset_data_dir.sh --spk-list ${line} data/tts/${x}_gt_rm_max data/tts/${x}_gt_rm_max_${spk}
            done
        done

    fi

    if [ ${tts_dump} == 'true' ]; then
        echo "Extract single speaker from data/tts/test_clean"
        nj=5
        # make a dev set
        utils/subset_data_dir.sh --last data/tts/${tts_data_set} ${deveval_num} data/tts/${tts_data_set}_deveval
        utils/subset_data_dir.sh --last data/tts/${tts_data_set}_deveval ${eval_num} data/tts/${eval_set}
        utils/subset_data_dir.sh --first data/tts/${tts_data_set}_deveval ${dev_num} data/tts/${dev_set}
        n=$(( $(wc -l < data/tts/${tts_data_set}/wav.scp) - ${deveval_num} ))
        utils/subset_data_dir.sh --first data/tts/${tts_data_set} ${n} data/tts/${train_set}

        #dump features for decoding
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
          data/tts/${train_set}/feats.scp ${tts_cmvn} exp/tts/dump_feats/${train_set} \
          ${tts_feat_tr_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
          data/tts/${dev_set}/feats.scp ${tts_cmvn} exp/tts/dump_feats/${dev_set} \
          ${tts_feat_dt_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
          data/tts/${eval_set}/feats.scp ${tts_cmvn} exp/tts/dump_feats/${eval_set} \
          ${tts_feat_ev_dir}
    fi

fi

echo "dictionary: ${tts_dict}"
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 5:(TTS) Dictionary and Json Data Preparation"
    
    mkdir -p data/tts/lang_1char
    cp ${tts_dict} data/tts/lang_1char/${tts_dict##*/}

        # make json labels
        data2json.sh --feat ${tts_feat_tr_dir}/feats.scp \
            data/tts/${train_set} ${tts_dict} > ${tts_feat_tr_dir}/data.json
        data2json.sh --feat ${tts_feat_dt_dir}/feats.scp \
            data/tts/${dev_set} ${tts_dict} > ${tts_feat_dt_dir}/data.json
        data2json.sh --feat ${tts_feat_ev_dir}/feats.scp \
            data/tts/${eval_set} ${tts_dict} > ${tts_feat_ev_dir}/data.json
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6:(TTS) x-vector extraction"
    nj=1
    # Make MFCCs and compute the energy-based VAD for each dataset
    mfccdir=mfcc
    vaddir=mfcc
    # for name in ${extract_set//-/_}; do # for all check
    for name in ${train_set} ${dev_set} ${eval_set}; do
        utils/copy_data_dir.sh data/tts/${name} data/tts/${name}_mfcc_16k
        utils/data/resample_data_dir.sh 16000 data/tts/${name}_mfcc_16k
        steps/make_mfcc.sh \
            --write-utt2num-frames true \
            --mfcc-config conf/mfcc.conf \
            --nj ${nj} --cmd "$train_cmd" \
            data/tts/${name}_mfcc_16k exp/tts/make_mfcc_16k ${mfccdir}
        utils/fix_data_dir.sh data/tts/${name}_mfcc_16k
        sid/compute_vad_decision.sh --nj ${nj} --cmd "$train_cmd" \
            data/tts/${name}_mfcc_16k exp/tts/make_vad ${vaddir}
        utils/fix_data_dir.sh data/tts/${name}_mfcc_16k
        echo $name
    done

    # Check pretrained model existence
    nnet_dir=exp/xvector/xvector_nnet_1a
    if [ ! -e ${nnet_dir} ]; then
        echo "X-vector model does not exist. Download pre-trained model."
        wget http://kaldi-asr.org/models/8/0008_sitw_v2_1a.tar.gz
        tar xvf 0008_sitw_v2_1a.tar.gz
        mv 0008_sitw_v2_1a/exp/xvector_nnet_1a $(dirname $nnet_dir)
        rm -rf 0008_sitw_v2_1a.tar.gz 0008_sitw_v2_1a
    fi
    
    # Extract x-vector
    # for name in ${extract_set//-/_}; do # for all check
    for name in ${train_set} ${dev_set} ${eval_set}; do
        sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj ${nj} \
            ${nnet_dir} data/tts/${name}_mfcc_16k \
            ${nnet_dir}/xvectors_${name}
    done
    
    # Update json
    # for name in ${extract_set//-/_}; do # for all check
    for name in ${train_set} ${dev_set} ${eval_set}; do
        local/update_json.sh ${dumpdir}/tts/${name}/data.json ${nnet_dir}/xvectors_${name}/xvector.scp
    done
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
[ ! -e ${tts_decode_config} ] && cat ${tts_pre_decode_config} > ${tts_decode_config}
[ ! -e ${expdir}/results/model.json ] && cp ${tts_model_conf} ${expdir}/results/model.json
pre_trained_tts_model=${expdir}/results/model.0th.copy
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7:(TTS) 0th Decoding"
    nj=5
    # for name in ${extract_set//-/_}; do # for all check
    for name in ${eval_set}; do #${dev_set} ${eval_set}; do
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
            --model ${pre_trained_tts_model} \
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


if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8:(TTS) 0th Synthesis"
    nj=5
    pids=() # initialize pids
    # for name in ${extract_set//-/_}; do # for all check
    for name in ${eval_set}; do #${dev_set} ${eval_set}; do
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

# outdir=${expdir}/outputs_${tts_model}_$(basename ${tts_decode_config%.*})_0th
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: Objective Evaluation"
    nj=5

    # ASR model selection for CER objective evaluation 
    asr_model_dir="exp/tts/${asr_eval_model}_asr"
    case "${asr_eval_model}" in
        "librispeech.transformer.ngpu4") asr_url="https://drive.google.com/open?id=1BtQvAnsFvVi-dp_qsaFP7n4A_5cwnlR6" \
          asr_cmvn="${asr_model_dir}/data/train_960/cmvn.ark" \
          asr_pre_decode_config="${asr_model_dir}/conf/tuning/decode_pytorch_transformer_large.yaml" \
          recog_model="${asr_model_dir}/exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug/results/model.val5.avg.best" \
          lang_model="${asr_model_dir}/exp/irielm.ep11.last5.avg/rnnlm.model.best" ;;

        *) echo "No such models: ${asr_eval_model}"; exit 1 ;;
    esac

    # ASR model download (librispeech)
    if [ ! -e ${asr_model_dir}/.complete ]; then
        mkdir -p ${asr_model_dir}
        download_from_google_drive.sh ${asr_url} ${asr_model_dir} ".tar.gz"
        touch ${asr_model_dir}/.complete
    fi
    echo "ASR model: ${asr_model_dir} exits."


    # Select TTS model
    if [ ${eval_tts_model} == 1 ]; then
        echo "Evaluate: $(basename ${train_config%.*})"
    else
        echo "Evaluate: ground truth"
        expdir=exp/ground_truth
        outdir=${expdir}/sym_link
        for name in ${dev_set} ${eval_set}; do
            mkdir -p ${outdir}_denorm/${name}/wav
            cat < data/${name}/wav.scp | awk '{print $1}' | while read -r filename; do
                if [ -L ${outdir}_denorm/${name}/wav/${filename}.wav ]; then
                    unlink ${outdir}_denorm/${name}/wav/${filename}.wav
                fi
                ln -s ${db_root}/LJSpeech-1.1/wavs/${filename}.wav ${outdir}_denorm/${name}/wav/${filename}.wav
            done
        done
    fi

    # Select decoder
    if [ ${voc} == "GL" ]; then
        dir_tail=""
    elif [ ${voc} == "WNV_softmax" ]; then
        dir_tail="_wnv_nsf"
    elif [ ${voc} == "WNV_mol" ]; then
        dir_tail="_wnv_mol"
    fi

    asr_data_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.data${dir_tail}"
    asr_fbank_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.fbank${dir_tail}"
    asr_feat_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.dump${dir_tail}"
    asr_result_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.result${dir_tail}"

    # Data preparation for ASR
    echo "8.1 Data preparation for ASR"
    for name in ${eval_set}; do #${dev_set} 
        local/data_prep_for_asr.sh ${outdir}_denorm/${name}/wav${dir_tail} ${asr_data_dir}/${name}
        cp data/tts/${name}/text ${asr_data_dir}/${name}/text
        utils/validate_data_dir.sh --no-feats ${asr_data_dir}/${name}
    done
    
    # Feature extraction for ASR
    echo "8.2 Feature extraction for ASR"
    for name in ${eval_set}; do #${dev_set} 
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} \
          --write_utt2num_frames true \
          --write_utt2dur false \
          ${asr_data_dir}/${name} \
          ${outdir}_denorm.ob_eval/${asr_model}_asr.make_fbank${dir_tail}/${name} \
          ${asr_fbank_dir}/${name}
        utils/fix_data_dir.sh ${asr_data_dir}/${name}

        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
          ${asr_data_dir}/${name}/feats.scp ${asr_cmvn} ${outdir}_denorm.ob_eval/${asr_model}_asr.dump_feats${dir_tail}/${name} \
          ${asr_feat_dir}/${name}
    done

    # Dictionary and Json Data Preparation
    echo "8.3 Dictionary and Json Data Preparation for ASR"
    asr_dict="data/tts/decode_dict/X.txt"; mkdir -p ${asr_dict%/*}
    echo "<unk> 1" > ${asr_dict}
    for name in ${eval_set}; do #${dev_set} 
        data2json.sh --feat ${asr_feat_dir}/${name}/feats.scp \
          ${asr_data_dir}/${name} ${asr_dict} > ${asr_feat_dir}/${name}/data.json
    done

    # ASR decoding
    echo "8.4 ASR decoding"
    asr_decode_config="conf/tuning/decode_asr.yaml"
    cat < ${asr_pre_decode_config} | sed -e 's/beam-size: 60/beam-size: 10/' > ${asr_decode_config}
    for name in ${eval_set}; do #${dev_set} 

        # split data
        splitjson.py --parts ${nj} ${asr_feat_dir}/${name}/data.json
    
        # set batchsize 0 to disable batch decoding    
        ${decode_cmd} JOB=1:${nj} ${asr_result_dir}.${api}/${name}/log/decode.JOB.log \
            asr_recog.py \
              --config ${asr_decode_config} \
              --ngpu 0 \
              --backend ${backend} \
              --batchsize 0 \
              --recog-json ${asr_feat_dir}/${name}/split${nj}utt/data.JOB.json \
              --result-label ${asr_result_dir}.${api}/${name}/data.JOB.json \
              --model ${recog_model} \
              --api ${api} \
              --rnnlm ${lang_model}

        score_sclite_wo_dict.sh --wer true ${asr_result_dir}.${api}/${name}

    done
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
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "stage 9:(TTS) Model Training"
    
    cat ${tts_pre_train_config} | sed -e 's/epochs: 100/epochs: 100/' \
    | sed -e 's/transformer-lr: 1.0/transformer-lr: 1.0/' \
    | sed -e 's/# other/# other\nreport-interval-iters: 1/' \
    | sed -e 's/accum-grad: 4/accum-grad: 1/' \
    | sed -e 's/dropout-rate: 0.1/dropout-rate: 0.1/g' \
    | sed -e 's/dropout-rate: 0.5/dropout-rate: 0.5/g' \
    > ${tts_train_config}
    
    
    tr_json=${tts_feat_tr_dir}/data.json
    dt_json=${tts_feat_dt_dir}/data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        tts_train.py \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --outdir ${expdir}/results \
           --tensorboard-dir tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${tts_resume} \
           --pre-trained-model ${pre_trained_tts_model} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${tts_train_config}
fi

if [ ${n_average} -gt 0 ]; then
    tts_model=model.last${n_average}.avg.best
fi
outdir=${expdir}/outputs_${tts_model}_$(basename ${tts_decode_config%.*})
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "stage 10:(TTS) Decoding"
    nj=5
    if [ ${n_average} -gt 0 ]; then
        average_checkpoints.py --backend ${backend} \
                               --snapshots ${expdir}/results/snapshot.ep.* \
                               --out ${expdir}/results/${tts_model} \
                               --num ${n_average}
    fi
    pids=() # initialize pids
    for name in ${eval_set}; do #${dev_set} ${eval_set}; do
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
                --model ${expdir}/results/${tts_model} \
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

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    echo "stage 11:(TTS) Synthesis"
    nj=5
    pids=() # initialize pids
    for name in ${eval_set}; do #${dev_set} ${eval_set}; do
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


# outdir=${expdir}/outputs_${tts_model}_$(basename ${tts_decode_config%.*})_0th
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    echo "stage 12: Objective Evaluation"
    nj=5

    # ASR model selection for CER objective evaluation 
    asr_model_dir="exp/tts/${asr_eval_model}_asr"
    case "${asr_eval_model}" in
        "librispeech.transformer.ngpu4") asr_url="https://drive.google.com/open?id=1BtQvAnsFvVi-dp_qsaFP7n4A_5cwnlR6" \
          asr_cmvn="${asr_model_dir}/data/train_960/cmvn.ark" \
          asr_pre_decode_config="${asr_model_dir}/conf/tuning/decode_pytorch_transformer_large.yaml" \
          recog_model="${asr_model_dir}/exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug/results/model.val5.avg.best" \
          lang_model="${asr_model_dir}/exp/irielm.ep11.last5.avg/rnnlm.model.best" ;;

        *) echo "No such models: ${asr_eval_model}"; exit 1 ;;
    esac

    # ASR model download (librispeech)
    if [ ! -e ${asr_model_dir}/.complete ]; then
        mkdir -p ${asr_model_dir}
        download_from_google_drive.sh ${asr_url} ${asr_model_dir} ".tar.gz"
        touch ${asr_model_dir}/.complete
    fi
    echo "ASR model: ${asr_model_dir} exits."


    # Select TTS model
    if [ ${eval_tts_model} == 1 ]; then
        echo "Evaluate: $(basename ${train_config%.*})"
    else
        echo "Evaluate: ground truth"
        expdir=exp/ground_truth
        outdir=${expdir}/sym_link
        for name in ${dev_set} ${eval_set}; do
            mkdir -p ${outdir}_denorm/${name}/wav
            cat < data/${name}/wav.scp | awk '{print $1}' | while read -r filename; do
                if [ -L ${outdir}_denorm/${name}/wav/${filename}.wav ]; then
                    unlink ${outdir}_denorm/${name}/wav/${filename}.wav
                fi
                ln -s ${db_root}/LJSpeech-1.1/wavs/${filename}.wav ${outdir}_denorm/${name}/wav/${filename}.wav
            done
        done
    fi

    # Select decoder
    if [ ${voc} == "GL" ]; then
        dir_tail=""
    elif [ ${voc} == "WNV_softmax" ]; then
        dir_tail="_wnv_nsf"
    elif [ ${voc} == "WNV_mol" ]; then
        dir_tail="_wnv_mol"
    fi

    asr_data_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.data${dir_tail}"
    asr_fbank_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.fbank${dir_tail}"
    asr_feat_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.dump${dir_tail}"
    asr_result_dir="${outdir}_denorm.ob_eval/${asr_model}_asr.result${dir_tail}"

    # Data preparation for ASR
    echo "12.1 Data preparation for ASR"
    for name in ${eval_set}; do #${dev_set} 
        local/data_prep_for_asr.sh ${outdir}_denorm/${name}/wav${dir_tail} ${asr_data_dir}/${name}
        cp data/tts/${name}/text ${asr_data_dir}/${name}/text
        utils/validate_data_dir.sh --no-feats ${asr_data_dir}/${name}
    done
    
    # Feature extraction for ASR
    echo "12.2 Feature extraction for ASR"
    for name in ${eval_set}; do #${dev_set} 
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} \
          --write_utt2num_frames true \
          --write_utt2dur false \
          ${asr_data_dir}/${name} \
          ${outdir}_denorm.ob_eval/${asr_model}_asr.make_fbank${dir_tail}/${name} \
          ${asr_fbank_dir}/${name}
        utils/fix_data_dir.sh ${asr_data_dir}/${name}

        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
          ${asr_data_dir}/${name}/feats.scp ${asr_cmvn} ${outdir}_denorm.ob_eval/${asr_model}_asr.dump_feats${dir_tail}/${name} \
          ${asr_feat_dir}/${name}
    done

    # Dictionary and Json Data Preparation
    echo "12.3 Dictionary and Json Data Preparation for ASR"
    asr_dict="data/tts/decode_dict/X.txt"; mkdir -p ${asr_dict%/*}
    echo "<unk> 1" > ${asr_dict}
    for name in ${eval_set}; do #${dev_set} 
        data2json.sh --feat ${asr_feat_dir}/${name}/feats.scp \
          ${asr_data_dir}/${name} ${asr_dict} > ${asr_feat_dir}/${name}/data.json
    done

    # ASR decoding
    echo "12.4 ASR decoding"
    asr_decode_config="conf/tuning/decode_asr.yaml"
    cat < ${asr_pre_decode_config} | sed -e 's/beam-size: 60/beam-size: 10/' > ${asr_decode_config}
    for name in ${eval_set}; do #${dev_set} 

        # split data
        splitjson.py --parts ${nj} ${asr_feat_dir}/${name}/data.json
    
        # set batchsize 0 to disable batch decoding    
        ${decode_cmd} JOB=1:${nj} ${asr_result_dir}.${api}/${name}/log/decode.JOB.log \
            asr_recog.py \
              --config ${asr_decode_config} \
              --ngpu 0 \
              --backend ${backend} \
              --batchsize 0 \
              --recog-json ${asr_feat_dir}/${name}/split${nj}utt/data.JOB.json \
              --result-label ${asr_result_dir}.${api}/${name}/data.JOB.json \
              --model ${recog_model} \
              --api ${api} \
              --rnnlm ${lang_model}

        score_sclite_wo_dict.sh --wer true ${asr_result_dir}.${api}/${name}

    done
fi
