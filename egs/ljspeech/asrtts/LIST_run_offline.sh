#!/bin/bash

# Copyright 2019 Okayama University (Katsuki Inoue)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

ASR_VC="test_clean_asr_no_0rec_22050"
GT_TTS="test_clean_adapt_no_0rec_22050"

# ----------------stage -2 ~ 4 ----------------
# ./run_offline.sh -2 -2 "1089" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(o),abelab(o)
# ./run_offline.sh  0  0 "1089" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(o),abelab(o)
# ./run_offline.sh  4  4 "1089" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(o),abelab(o)


# ----------------stage 4 ~ 5 ----------------
# Data preparation for last half (ASR-text)
# ./run_offline.sh 4 5 "1089" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(o),abelab(o)
# ./run_offline.sh 4 5 "2300"  7  7 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(o),abelab(o)
# ./run_offline.sh 4 5 "8230"  6  6 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(o),abelab(o)
# ./run_offline.sh 4 5 "237"  15 15 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(o),abelab(o)
# ./run_offline.sh 4 5 "4446" 18 18 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(o),abelab(o)
# ./run_offline.sh 4 5 "5683" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(o),abelab(o)

# Data preparation for last half (GT-text)
# ./run_offline.sh 4 5 "1089" 10 10 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(o)
# ./run_offline.sh 4 5 "2300"  7  7 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(o)
# ./run_offline.sh 4 5 "8230"  6  6 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(o)
# ./run_offline.sh 4 5 "237"  15 15 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(o)
# ./run_offline.sh 4 5 "4446" 18 18 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(o)
# ./run_offline.sh 4 5 "5683" 10 10 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(o)


# ----------------stage 6 ~ 7 ----------------
# 0th decode (GT-text:TTS)
# ./run_offline.sh 6 7 "1089" 10 10  ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
# ./run_offline.sh 6 7 "2300"  7  7  ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
# ./run_offline.sh 6 7 "8230"  6  6  ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
# ./run_offline.sh 6 7 "237"  15 15  ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
# ./run_offline.sh 6 7 "4446" 18 18  ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
# ./run_offline.sh 6 7 "5683" 10 10  ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)

# 0th decode (ASR-text:VC)
# ./run_offline.sh 6 7 "1089" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(o)
# ./run_offline.sh 6 7 "2300"  7  7 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(o)
# ./run_offline.sh 6 7 "8230"  6  6 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(o)
# ./run_offline.sh 6 7 "237"  15 15 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(o)
# ./run_offline.sh 6 7 "4446" 18 18 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(o)
# ./run_offline.sh 6 7 "5683" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(o)


# ----------------stage 8 ~ 10 ----------------
# ASR train & Decode for VC eval
# ./run_offline.sh 8 10 "1089" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(o),abelab(x)
# ./run_offline.sh 8 10 "2300"  7  7 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(o),abelab(x)
# ./run_offline.sh 8 10 "8230"  6  6 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(o),abelab(x)
# ./run_offline.sh 8 10 "237"  15 15 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(o),abelab(x)
# ./run_offline.sh 8 10 "4446" 18 18 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(o),abelab(x)
# ./run_offline.sh 8 10 "5683" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(o),abelab(x)

# GT train & Decode for TTS eval
# ./run_offline.sh 8 10 "1089" 10 10 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(x)
# ./run_offline.sh 8 10 "2300"  7  7 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(x)
# ./run_offline.sh 8 10 "8230"  6  6 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(x)
# ./run_offline.sh 8 10 "237"  15 15 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(x)
# ./run_offline.sh 8 10 "4446" 18 18 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(x)
# ./run_offline.sh 8 10 "5683" 10 10 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(x)


# ----------------stage 9 ~ 10 ----------------
# Decode for TTS eval
./run_offline.sh 9 10 "1089" 10 10 ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
./run_offline.sh 9 10 "2300"  7  7 ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
./run_offline.sh 9 10 "8230"  6  6 ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
./run_offline.sh 9 10 "237"  15 15 ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
./run_offline.sh 9 10 "4446" 18 18 ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
./run_offline.sh 9 10 "5683" 10 10 ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)


# ----------------stage 11 ----------------
# 0th decode (GT-text:TTS)
# ./run_offline.sh 11 11 "1089" 10 10  ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(xo)
# ./run_offline.sh 11 11 "2300"  7  7  ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(xo)
# ./run_offline.sh 11 11 "8230"  6  6  ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(xo)
# ./run_offline.sh 11 11 "237"  15 15  ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(xo)
# ./run_offline.sh 11 11 "4446" 18 18  ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(xo)
# ./run_offline.sh 11 11 "5683" 10 10  ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(xo)

# 0th decode (ASR-text:VC)
# ./run_offline.sh 11 11 "1089" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(xo)
# ./run_offline.sh 11 11 "2300"  7  7 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(xo)
# ./run_offline.sh 11 11 "8230"  6  6 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(xo)
# ./run_offline.sh 11 11 "237"  15 15 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(xo)
# ./run_offline.sh 11 11 "4446" 18 18 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(xo)
# ./run_offline.sh 11 11 "5683" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(xo)

# WNV decode
# ./run_offline.sh 11 11 "1089" 10 10 ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
# ./run_offline.sh 11 11 "2300"  7  7 ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
# ./run_offline.sh 11 11 "8230"  6  6 ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
# ./run_offline.sh 11 11 "237"  15 15 ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
# ./run_offline.sh 11 11 "4446" 18 18 ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
# ./run_offline.sh 11 11 "5683" 10 10 ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)

# ./run_offline.sh 11 11 "1089" 10 10 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(x)
# ./run_offline.sh 11 11 "2300"  7  7 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(x)
# ./run_offline.sh 11 11 "8230"  6  6 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(x)
# ./run_offline.sh 11 11 "237"  15 15 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(x)
# ./run_offline.sh 11 11 "4446" 18 18 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(x)
# ./run_offline.sh 11 11 "5683" 10 10 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(x)


# ----------------stage 12 ----------------
# 0th decode (GT-text:TTS)
# ./run_offline.sh 12 12 "1089" 10 10  ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
# ./run_offline.sh 12 12 "2300"  7  7  ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
# ./run_offline.sh 12 12 "8230"  6  6  ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
# ./run_offline.sh 12 12 "237"  15 15  ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
# ./run_offline.sh 12 12 "4446" 18 18  ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
# ./run_offline.sh 12 12 "5683" 10 10  ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)

# 0th decode (ASR-text:VC)
# ./run_offline.sh 12 12 "1089" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(o)
# ./run_offline.sh 12 12 "2300"  7  7 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(o)
# ./run_offline.sh 12 12 "8230"  6  6 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(o)
# ./run_offline.sh 12 12 "237"  15 15 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(o)
# ./run_offline.sh 12 12 "4446" 18 18 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(o)
# ./run_offline.sh 12 12 "5683" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(o)

# TTS eval (GT-text:TTS)
# ./run_offline.sh 12 12 "1089" 10 10 ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
# ./run_offline.sh 12 12 "2300"  7  7 ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
# ./run_offline.sh 12 12 "8230"  6  6 ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
# ./run_offline.sh 12 12 "237"  15 15 ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
# ./run_offline.sh 12 12 "4446" 18 18 ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)
# ./run_offline.sh 12 12 "5683" 10 10 ${ASR_VC} ${GT_TTS} ${GT_TTS} #@ gss(x),abelab(o)

# ./run_offline.sh 12 12 "1089" 10 10 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(x)
# ./run_offline.sh 12 12 "2300"  7  7 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(x)
# ./run_offline.sh 12 12 "8230"  6  6 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(x)
# ./run_offline.sh 12 12 "237"  15 15 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(x)
# ./run_offline.sh 12 12 "4446" 18 18 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(x)
# ./run_offline.sh 12 12 "5683" 10 10 ${GT_TTS} ${GT_TTS} ${GT_TTS} #@ gss(o),abelab(x)

# VC eval (ASR-text:VC)
# ./run_offline.sh 12 12 "1089" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(o),abelab(x)
# ./run_offline.sh 12 12 "2300"  7  7 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(o),abelab(x)
# ./run_offline.sh 12 12 "8230"  6  6 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(o),abelab(x)
# ./run_offline.sh 12 12 "237"  15 15 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(o),abelab(x)
# ./run_offline.sh 12 12 "4446" 18 18 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(o),abelab(x)
# ./run_offline.sh 12 12 "5683" 10 10 ${ASR_VC} ${ASR_VC} ${ASR_VC} #@ gss(o),abelab(x)

# ./run_offline.sh 12 12 "1089" 10 10 ${GT_TTS} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(x)
# ./run_offline.sh 12 12 "2300"  7  7 ${GT_TTS} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(x)
# ./run_offline.sh 12 12 "8230"  6  6 ${GT_TTS} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(x)
# ./run_offline.sh 12 12 "237"  15 15 ${GT_TTS} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(x)
# ./run_offline.sh 12 12 "4446" 18 18 ${GT_TTS} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(x)
# ./run_offline.sh 12 12 "5683" 10 10 ${GT_TTS} ${ASR_VC} ${ASR_VC} #@ gss(x),abelab(x)
