#!/bin/bash

split='devtest'

python3 ./format_task1.py \
 --step1-pred-txt ./data/prediction/ \
 --split-path ./data/simmc2_dials_dstc10_${split}.json \
 --save-path  ./data/prediction/dstc10-simmc-$split-pred-subtask-1.json \

python3 ./run_devtest_preprocess.py \
 --pred-txt ./data/prediction/step1_${split}_predict_results.txt \
 --target-txt ./data/finetuning_data/chunk/${split}.txt \
 --slot-mapping-path ./data/models/electra_large/finetuning_tfrecords/chunk_tfrecords/chunk_slot_types.pkl \
 --action-mapping-path ./data/models/electra_large/finetuning_tfrecords/chunk_tfrecords/chunk_action.pkl \
 --split $split\
 --save-path ../data/finetuning_data/chunk/${split}.txt
