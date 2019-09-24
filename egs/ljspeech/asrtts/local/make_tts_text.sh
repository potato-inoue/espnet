#!/bin/bash

# Copyright 2019 Okayama University (Katsuki Inoue)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

asr_result_dir=$1
tts_data_dir=$2

set -euo pipefail

# make tmp_txt from rec_text
grep rec_text ${asr_result_dir}/result.json \
  | sed -e 's/.*: "\(.*\)".*/\1/' \
  | sed -e 's/▁//' -e 's/<eos>//' \
  | tr a-z A-Z \
  | awk '{print toupper(substr($0,0,1))substr($0,2,length($0))"."}' \
  | sed -e 's/▁/ /g' \
  > local/tmp_text

# make tmp_utt from original text
cat ${tts_data_dir}/text | awk '{print $1}' > local/tmp_utt

# rename original text
if [ ! -e {tts_data_dir}/text_org ]; then
  mv ${tts_data_dir}/text ${tts_data_dir}/text_org
fi

# make new text
paste -d " " local/tmp_utt local/tmp_text > ${tts_data_dir}/text

# remove tmp files
rm local/tmp_utt
rm local/tmp_text
