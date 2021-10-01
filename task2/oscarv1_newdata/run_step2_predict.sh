#!/bin/bash

split='devtest'
python3 ./format_task2.py \
 --step2-bin  resultfromsys.binupdate \
 --split-path ../../task3/data/simmc2_dials_dstc10_${split}.json \
 --save-path  ../../results/dstc10-simmc-${split}-pred-subtask-2.json
