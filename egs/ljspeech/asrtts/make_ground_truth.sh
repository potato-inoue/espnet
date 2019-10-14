#!/bin/bash

# Copyright 2019 Okayama University (Katsuki Inoue)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

ex_set="test_clean"
for spk in "237" "2300" "4446" "5683" "8230"; do

    # dir="exp/tts/${ex_set}_22050_${spk}_train_no_dev_pytorch_tts_train_pytorch_transformer.fine-tuning.spk${spk}_lr1.rev1e-1/outputs_model.last1.avg.best_tts_decode.fine-tuning_denorm/${ex_set}_22050_${spk}_eval/wav/"
    dir="exp/tts/${ex_set}_asr_no_0rec_22050_${spk}_train_no_dev_pytorch_tts_train_pytorch_transformer.fine-tuning.spk${spk}_lr1e-1.rev1/outputs_model.last1.avg.best_tts_decode.fine-tuning_denorm/test_clean_adapt_no_0rec_22050_${spk}_eval/wav/"

    # out_dir="exp/tts/all_train_pytorch_tts_train_pytorch_transformer.fine-tuning.all/outputs_libritts.transformer.v1_tts_decode.fine-tuning_0th_denorm/${ex_set}_ex/wav/"
    out_dir="exp/tts/ground_truth/"
    mkdir -p ${out_dir}


    find ${dir} -name "*.wav" | sort | while read line; do
        new_spk_id=${spk}
        new_spk_book=$(echo ${line##*/} | awk -F'_' '{printf("%s\n",$2)}')

        # cp /export/a06/katsuki/DB/LibriTTS/${ex_set//_/-}/${new_spk_id}/${new_spk_book}/${line##*/} ${out_dir}/
        cp /abelab/DB4/LibriTTS/${ex_set//_/-}/${new_spk_id}/${new_spk_book}/${line##*/} ${out_dir}/
    done

done
