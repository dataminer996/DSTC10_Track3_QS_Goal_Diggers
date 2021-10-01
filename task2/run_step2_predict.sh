#!/bin/bash

split='devtest'
python3 ./format_task2.py \
 --step2-bin  oscarv1_newdata/modelgood/find_0822_lowercase_lrnew_5e-05batch3weight3.7gpu6_epoch_20/checkpoint-19-11580/resultfromsys.binupdate \
 --split-path ../task3/data/simmc2_dials_dstc10_${split}.json \
 --save-path ../results/dstc10-simmc-${split}-pred-subtask-2.json
