#!/bin/bash

# Copyright 2019 Okayama University (Katsuki Inoue)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

ex_set="dev_clean"

dir="exp/tts/all_train_pytorch_tts_train_pytorch_transformer.fine-tuning.all/outputs_libritts.transformer.v1_tts_decode.fine-tuning_0th_denorm/${ex_set}/wav/"

# out_dir="exp/tts/all_train_pytorch_tts_train_pytorch_transformer.fine-tuning.all/outputs_libritts.transformer.v1_tts_decode.fine-tuning_0th_denorm/${ex_set}_ex/wav/"
out_dir="exp/tts/ground_truth/${ex_set}_ex/wav/"
mkdir -p ${out_dir}

now_spk="aaa"

find ${dir} -name "*.wav" | sort | while read line; do
    new_spk=$(echo ${line##*/} | awk -F'_' '{printf("%s_%s\n",$1,$2)}')
    new_spk_id=$(echo ${line##*/} | awk -F'_' '{printf("%s\n",$1)}')
    new_spk_book=$(echo ${line##*/} | awk -F'_' '{printf("%s\n",$2)}')

    if [ ${now_spk} != ${new_spk} ]; then
        now_spk=${new_spk}
        echo ${now_spk}
        # cp ${line} ${out_dir}

        cp  /export/a06/katsuki/DB/LibriTTS/${ex_set//_/-}/${new_spk_id}/${new_spk_book}/${line##*/} ${out_dir}
    fi
done
