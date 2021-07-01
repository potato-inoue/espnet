#!/bin/bash

# db=/work/abelab/DB/LJspeech-1.1/wavs

for data_set in char_train char_dev char_eval; do

 echo ${data_set}

 out_file=data/${data_set}/dur.txt
 echo -n > ${out_file}

 cat data/${data_set}/wav.scp | while read line; do
  wav=`echo ${line} | awk '{print $2}'`
  dur=`soxi -D ${wav}`
  echo ${wav},${dur} >> ${out_file}
 done

done

