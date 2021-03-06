#!/bin/bash


DATA_PATH='../data'
STEP1_SAVE_PATH='../data/finetuning_data/chunk/'
SPLIT='devtest'
python3 run_preprocess.py \
 --simmc_json $DATA_PATH/train_sobject.json \
 --split train \
 --action-save-path $STEP1_SAVE_PATH \

python3 run_preprocess.py \
 --simmc_json $DATA_PATH/dev_sobject.json \
 --split dev \
 --action-save-path $STEP1_SAVE_PATH \

python3 run_preprocess.py \
 --simmc_json $DATA_PATH/devtest_sobject.json \
 --split $SPLIT \
 --action-save-path $STEP1_SAVE_PATH \
