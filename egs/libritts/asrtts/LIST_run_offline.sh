#!/bin/bash

# Copyright 2019 Okayama University (Katsuki Inoue)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# run_offline.sh 1e0.rev0
# run_offline.sh 1e0.rev1
# run_offline.sh 1e0.rev2
# run_offline.sh 1e0.rev3
# run_offline.sh 1e0.rev4
# run_offline.sh 1e0.rev5
# run_offline.sh 1e0.rev6
# run_offline.sh 1e0.rev7
# run_offline.sh 1e0.rev8
# run_offline.sh 1e0.rev9
# run_offline.sh 1e0.rev10
# run_offline.sh 1e-1.rev1
# run_offline.sh 1e-1.rev2
# run_offline.sh 1e-1.rev3
# run_offline.sh 1e-2.rev1
# run_offline.sh 1e-3.rev1

ASR_VC="test_clean_asr_rm_max"
GT_TTS="test_clean_gt_rm_max"

# Data preparation for last half (ASR-text)
# ./run_offline.sh 4 5 "1089" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 4 5 "2300"  7  7 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 4 5 "8230"  6  6 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 4 5 "237"  15 15 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 4 5 "4446" 18 18 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 4 5 "5683" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# Data preparation for last half (GT-text)
# ./run_offline.sh 4 5 "1089" 10 10 ${GT_TTS} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 4 5 "2300"  7  7 ${GT_TTS} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 4 5 "8230"  6  6 ${GT_TTS} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 4 5 "237"  15 15 ${GT_TTS} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 4 5 "4446" 18 18 ${GT_TTS} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 4 5 "5683" 10 10 ${GT_TTS} ${GT_TTS} ${GT_TTS}

# Data preparation for last half (ASR-text)
# ./run_offline.sh 6 6 "1089" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 6 6 "2300"  7  7 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 6 6 "8230"  6  6 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 6 6 "237"  15 15 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 6 6 "4446" 18 18 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 6 6 "5683" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# Data preparation for last half (GT-text)
# ./run_offline.sh 6 6 "1089" 10 10 ${GT_TTS} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 6 6 "2300"  7  7 ${GT_TTS} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 6 6 "8230"  6  6 ${GT_TTS} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 6 6 "237"  15 15 ${GT_TTS} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 6 6 "4446" 18 18 ${GT_TTS} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 6 6 "5683" 10 10 ${GT_TTS} ${GT_TTS} ${GT_TTS}


# ./run_offline.sh 7 8 "2300"  7  7 ${ASR_VC} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 7 8 "8230"  6  6 ${ASR_VC} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 7 8 "237"  15 15 ${ASR_VC} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 7 8 "4446" 18 18 ${ASR_VC} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 7 8 "5683" 10 10 ${ASR_VC} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 7 8 "1089" 10 10 ${ASR_VC} ${GT_TTS} ${GT_TTS}
# ASR-train & ASR-Decode (ASR-text:VC)
# ./run_offline.sh 7 8 "2300"  7  7 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 7 8 "8230"  6  6 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 7 8 "237"  15 15 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 7 8 "4446" 18 18 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 7 8 "5683" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 7 8 "1089" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC}

# ./run_offline.sh 8 8 "2300"  7  7 ${ASR_VC} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 8 8 "8230"  6  6 ${ASR_VC} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 8 8 "237"  15 15 ${ASR_VC} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 8 8 "4446" 18 18 ${ASR_VC} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 8 8 "5683" 10 10 ${ASR_VC} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 8 8 "1089" 10 10 ${ASR_VC} ${GT_TTS} ${GT_TTS}
# ASR-train & ASR-Decode (ASR-text:VC)
# ./run_offline.sh 8 8 "2300"  7  7 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 8 8 "8230"  6  6 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 8 8 "237"  15 15 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 8 8 "4446" 18 18 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 8 8 "5683" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 8 8 "1089" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC}

# ./run_offline.sh 9 12 "2300"  7  7 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 9 12 "8230"  6  6 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 9 12 "237"  15 15 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 9 12 "4446" 18 18 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 9 12 "5683" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC}
# ./run_offline.sh 9 12 "1089" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC}


# ./run_offline.sh 13 13 "2300"  7  7 ${ASR_VC} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 13 13 "8230"  6  6 ${ASR_VC} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 13 13 "237"  15 15 ${ASR_VC} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 13 13 "4446" 18 18 ${ASR_VC} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 13 13 "5683" 10 10 ${ASR_VC} ${GT_TTS} ${GT_TTS}
# ./run_offline.sh 13 13 "1089" 10 10 ${ASR_VC} ${GT_TTS} ${GT_TTS}

./run_offline.sh 13 13 "2300"  7  7 ${ASR_VC} ${ASR_VC} ${ASR_VC}
./run_offline.sh 13 13 "8230"  6  6 ${ASR_VC} ${ASR_VC} ${ASR_VC}
./run_offline.sh 13 13 "237"  15 15 ${ASR_VC} ${ASR_VC} ${ASR_VC}
./run_offline.sh 13 13 "4446" 18 18 ${ASR_VC} ${ASR_VC} ${ASR_VC}
./run_offline.sh 13 13 "5683" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC}
./run_offline.sh 13 13 "1089" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC}
