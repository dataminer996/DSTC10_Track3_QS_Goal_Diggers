#!/bin/bash

split='devtest'
python3 ./format_task4_generation.py \
 --generation-pred-txt  oscargen_noobj_newdata/modelgood/find_0822lrnew_0.00017batch18weight3gpu1_epoch_242/checkpoint-241-42592/resulefile.txt \
 --split-path  ../generate/generagefeature/data/simmc2_dials_dstc10_${split}.json \
 --save-path  ./result/dstc10-simmc-${split}-pred-subtask-4-generation.json
