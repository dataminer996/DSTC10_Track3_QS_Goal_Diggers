#!/bin/bash


DATA_PATH='../data'
STEP1_SAVE_PATH='../data/finetuning_data/chunk/'
STEP3_SAVE_PATH='../../task3/data/finetuning_data/chunk/'
SPLIT='teststd'

python3 run_preprocess.py \
 --simmc_json $DATA_PATH/simmc2_dials_dstc10_teststd.json \
 --split $SPLIT \
 --action-save-path $STEP1_SAVE_PATH 
