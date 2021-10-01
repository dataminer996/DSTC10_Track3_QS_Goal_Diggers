#!/bin/bash

split='devtest'
python3 ./format_task4_retrieval_0923.py \
 --dir-bin oscar_retri_v1/0918/find_0822_lowercase_lrnew_5e-05batch3weight2.0gpu6_epoch_15/checkpoint-14-10575/resultfromsys.bin \
 --split-path ../generate/generagefeature/data/simmc2_dials_dstc10_${split}_retrieval_candidates.json \
 --save-path  dstc10-simmc-${split}-pred-subtask-4-retrieval.json 
