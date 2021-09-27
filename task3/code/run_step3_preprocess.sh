#!/bin/bash

DATA_DIR='../data'
STEP1_PATH='../../task1'
python3 run_preprocess.py \
  --simmc_train_json $DATA_DIR/simmc2_dials_dstc10_train.json \
  --simmc_dev_json $DATA_DIR/simmc2_dials_dstc10_dev.json \
  --simmc_devtest_json $DATA_DIR/simmc2_dials_dstc10_devtest.json \
  --train-target-txt $STEP1_PATH/data/finetuning_data/chunk/train.txt \
  --dev-target-txt $STEP1_PATH/data/finetuning_data/chunk/dev.txt \
  --devtest-target-txt $STEP1_PATH/data/finetuning_data/chunk/devtest.txt \
  --step1-best-pred $STEP1_PATH/result/prediction/step1_devtest_predict_results.txt \
  --save-path $DATA_DIR/finetuning_data/chunk \
  --slot-candidate $DATA_DIR/slot_candidate.json \
  --slot-mapping $DATA_DIR/finetuning_tfrecords/chunk_tfrecords/chunk_slot_types.pkl \
  --action-mapping $DATA_DIR/finetuning_tfrecords/chunk_tfrecords/chunk_action.pkl \
  --step1-action $STEP1_PATH/result/devtest_action_embedding.json
