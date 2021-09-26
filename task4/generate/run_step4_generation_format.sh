#!/bin/bash

split='devtest'
python3 ./format_task4_generation.py \
 --generation-pred-txt  oscargen_noobj_newdata/modelgood/checkpoint-241-42592/resulefile.txt \
 --split-path  /ceph/dstc/track3/allcode/task1/data/simmc2_dials_dstc10_${split}.json \
 --save-path  ./dstc10-simmc-${split}-pred-subtask-4-generation.json
