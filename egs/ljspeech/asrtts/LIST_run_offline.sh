#!/bin/bash

# Copyright 2019 Okayama University (Katsuki Inoue)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Dare preparation
./run_offline.sh 4 5 "1089" 10 10 
./run_offline.sh 4 5 "2300" 7 7 
./run_offline.sh 4 5 "8230" 6 6 
./run_offline.sh 4 5 "237" 15 15 
./run_offline.sh 4 5 "4446" 18 18 
./run_offline.sh 4 5 "5683" 10 10 

# 0th decode
# ./run_offline.sh 4 5 "1089" 10 10 
./run_offline.sh 4 5 "2300" 7 7 
./run_offline.sh 4 5 "8230" 6 6 
# ./run_offline.sh 4 5 "237" 15 15 
# ./run_offline.sh 4 5 "4446" 18 18 
./run_offline.sh 4 5 "5683" 10 10 

# TTS train & decode
# ./run_offline.sh 8 10 "1089" 10 10 
# ./run_offline.sh 8 10 "8230" 6 6 
# ./run_offline.sh 8 10 "237" 15 15 
# ./run_offline.sh 8 10 "4446" 18 18 
# ./run_offline.sh 8 10 "5683" 10 10 

# decode
# ./run_offline.sh 9 10 "2300" 7 7 
# ./run_offline.sh 9 10 "8230" 6 6 

# WNV decode
# ./run_offline.sh 11 11 "1089" 10 10 
# ./run_offline.sh 11 11 "8230" 6 6 
# ./run_offline.sh 11 11 "237" 15 15 
# ./run_offline.sh 11 11 "4446" 18 18 
# ./run_offline.sh 11 11 "5683" 10 10 
