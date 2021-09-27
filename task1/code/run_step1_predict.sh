#!/bin/bash

SPLIT='devtest'
SAVE_PATH='../result'
DATA_DIR='../data'
PRED_DIR='../result/prediction/'

python3 ./format_task1.py \
 --step1-pred-txt $PRED_DIR \
 --split-path $DATA_DIR/simmc2_dials_dstc10_${SPLIT}.json \
 --save-path  $SAVE_PATH/dstc10-simmc-$SPLIT-pred-subtask-1.json \
 --save-path-for-other-step $SAVE_PATH \
 --do-predict-split $SPLIT

