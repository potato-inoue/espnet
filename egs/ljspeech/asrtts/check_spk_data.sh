#!/bin/bash

echo -e -n "spk\tall\tpair\tasr\t"
echo -e -n "p_tr\tnp_tr\tdev\teval\t"
echo -e "p_tr\tnp_tr\tdev\teval\t"

for spk in 1089 2300 8230 237 4446 5683; do

    # stage 0
    # make dir
    mkdir -p data_check/


    # stage 1
    # get data-set info about utt
    grep ${spk}_ data/tts/test_clean_org_24000/wav.scp | awk '{print $1}' > data_check/${spk}.org.list
    grep ${spk}_ data/tts/test_clean_adapt_no_0rec_22050/wav.scp | awk '{print $1}' > data_check/${spk}.adapt.list
    grep ${spk}_ data/tts/test_clean_asr_no_0rec_22050/wav.scp | awk '{print $1}' > data_check/${spk}.asr.list

    org_num=$(cat data_check/${spk}.org.list | wc -l)
    full_num=$(cat data_check/${spk}.adapt.list | wc -l)
    asr_num=$(cat data_check/${spk}.asr.list | wc -l)

    # get train-set info about utt
    grep ${spk}_ data/tts/test_clean_adapt_no_0rec_22050_${spk}_train_no_dev/wav.scp | awk '{print $1}' > data_check/${spk}.adapt.tr.list
    p_tr_utt=$(cat data_check/${spk}.adapt.tr.list | wc -l)
    grep ${spk}_ data/tts/test_clean_asr_no_0rec_22050_${spk}_train_no_dev/wav.scp | awk '{print $1}' > data_check/${spk}.asr.tr.list
    np_tr_utt=$(cat data_check/${spk}.asr.tr.list | wc -l)
    
    # get dev-set info about utt
    grep ${spk}_ data/tts/test_clean_adapt_no_0rec_22050_${spk}_dev/wav.scp | awk '{print $1}' > data_check/${spk}.adapt.dev.list
    grep ${spk}_ data/tts/test_clean_asr_no_0rec_22050_${spk}_dev/wav.scp | awk '{print $1}' > data_check/${spk}.asr.dev.list
    diff data_check/${spk}.adapt.dev.list data_check/${spk}.asr.dev.list
    dev_utt=$(cat data_check/${spk}.adapt.dev.list | wc -l)

    # get eval-set info about utt
    grep ${spk}_ data/tts/test_clean_adapt_no_0rec_22050_${spk}_eval/wav.scp | awk '{print $1}' > data_check/${spk}.adapt.eval.list
    grep ${spk}_ data/tts/test_clean_asr_no_0rec_22050_${spk}_eval/wav.scp | awk '{print $1}' > data_check/${spk}.asr.eval.list
    diff data_check/${spk}.adapt.eval.list data_check/${spk}.asr.eval.list
    eval_utt=$(cat data_check/${spk}.adapt.eval.list | wc -l)

    # stage 2
    # check dur of p_tr
    echo -n > data_check/${spk}.adapt.tr.dur
    cat data_check/${spk}.adapt.tr.list | while read -r line; do
        book=$(echo ${line} | awk -F_ '{print $2}')
        sox --i -D /work/abelab/DB/LibriTTS/test-clean/${spk}/${book}/${line}.wav >> data_check/${spk}.adapt.tr.dur
    done
    p_tr_dur=$(cat data_check/${spk}.adapt.tr.dur | awk '{sum=sum+$1}END{print sum/60}')

    # check dur of np_tr
    echo -n > data_check/${spk}.asr.tr.dur
    cat data_check/${spk}.asr.tr.list | while read -r line; do
        book=$(echo ${line} | awk -F_ '{print $2}')
        sox --i -D /work/abelab/DB/LibriTTS/test-clean/${spk}/${book}/${line}.wav >> data_check/${spk}.asr.tr.dur
    done
    np_tr_dur=$(cat data_check/${spk}.asr.tr.dur | awk '{sum=sum+$1}END{print sum/60}')

    # check dur of dev
    echo -n > data_check/${spk}.asr.dev.dur
    cat data_check/${spk}.asr.dev.list | while read -r line; do
        book=$(echo ${line} | awk -F_ '{print $2}')
        sox --i -D /work/abelab/DB/LibriTTS/test-clean/${spk}/${book}/${line}.wav >> data_check/${spk}.asr.dev.dur
    done
    dev_dur=$(cat data_check/${spk}.asr.dev.dur | awk '{sum=sum+$1}END{print sum/60}')

    # check dur of eval
    echo -n > data_check/${spk}.asr.eval.dur
    cat data_check/${spk}.asr.eval.list | while read -r line; do
        book=$(echo ${line} | awk -F_ '{print $2}')
        sox --i -D /work/abelab/DB/LibriTTS/test-clean/${spk}/${book}/${line}.wav >> data_check/${spk}.asr.eval.dur
    done
    eval_dur=$(cat data_check/${spk}.asr.eval.dur | awk '{sum=sum+$1}END{print sum/60}')

    echo -e -n "${spk}\t${org_num}\t${full_num}\t${asr_num}\t"
    echo -e -n "${p_tr_utt}\t${np_tr_utt}\t${dev_utt}\t${eval_utt}\t"
    echo -e "${p_tr_dur}\t${np_tr_dur}\t${dev_dur}\t${eval_dur}\t"

done
