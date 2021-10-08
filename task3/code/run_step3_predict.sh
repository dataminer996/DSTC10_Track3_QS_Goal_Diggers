#!/bin/bash

SPLIT='devtest'
DATA_DIR='../data'
MODEL_DIR='../model'
RESULT_DIR='../result'
python3 ./format_task3.py \
 --action-result ../../task1/result/devtest_action_embedding.json \
 --step2-result ../../task2/results/dstc10-simmc-${SPLIT}-pred-subtask-2.json \
 --step3-pred-txt $RESULT_DIR/prediction_$SPLIT \
 --step3-traget-txt $DATA_DIR/finetuning_data/chunk/${SPLIT}.txt \
 --slot-mapping-path $DATA_DIR/finetuning_tfrecords/chunk_tfrecords/chunk_slot_types.pkl \
 --action-mapping-path $DATA_DIR/finetuning_tfrecords/chunk_tfrecords/chunk_action.pkl \
 --metadata-path $DATA_DIR \
 --scene-path /data_ssd/qinghua/simmc2/data/public \
 --split-path $DATA_DIR/simmc2_dials_dstc10_${SPLIT}.json \
 --save-path $RESULT_DIR/dstc10-simmc-${SPLIT}-pred-subtask-3.json

