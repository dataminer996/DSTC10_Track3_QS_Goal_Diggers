#!/bin/bash

split='devtest'
python3 evaluate_dst.py \
    --input_path_target  ../../task3/data/simmc2_dials_dstc10_${split}.json \
    --input_path_predicted ../../results/dstc10-simmc-${split}-pred-subtask-2.json \
    --output_path_report ./task2_eval_result.json

