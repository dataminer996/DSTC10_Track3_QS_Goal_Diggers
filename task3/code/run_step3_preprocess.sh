#!/bin/bash
SPLIT='devtest'
DATA_DIR='../data'
STEP1_PATH='../../task1'
python3 run_preprocess.py \
  --split train \
  --simmc_json $DATA_DIR/simmc2_dials_dstc10_train.json \
  --target-txt $STEP1_PATH/data/finetuning_data/chunk/train.txt \
  --step1-best-pred $STEP1_PATH/result/prediction_$SPLIT/epoch_5.0_lr_3e-05_ac_0.7_diam_0.7_fr_sys_0.7_ob_num_1.0_sl_0.7_cx_5e-05_sg_2.1e-11_2021-09-23-05-58-14_pred_result.txt \
  --save-path $DATA_DIR/finetuning_data/chunk \
  --slot-candidate $DATA_DIR/slot_candidate.json \
  --slot-mapping $DATA_DIR/finetuning_tfrecords/chunk_tfrecords/chunk_slot_types.pkl \
  --action-mapping $DATA_DIR/finetuning_tfrecords/chunk_tfrecords/chunk_action.pkl \
  --step1-action $STEP1_PATH/result/devtest_action_embedding.json

python3 run_preprocess.py \
  --split dev \
  --simmc_json $DATA_DIR/simmc2_dials_dstc10_dev.json \
  --target-txt $STEP1_PATH/data/finetuning_data/chunk/dev.txt \
  --step1-best-pred $STEP1_PATH/result/prediction_$SPLIT/epoch_5.0_lr_3e-05_ac_0.7_diam_0.7_fr_sys_0.7_ob_num_1.0_sl_0.7_cx_5e-05_sg_2.1e-11_2021-09-23-05-58-14_pred_result.txt \
  --save-path $DATA_DIR/finetuning_data/chunk \
  --slot-candidate $DATA_DIR/slot_candidate.json \
  --slot-mapping $DATA_DIR/finetuning_tfrecords/chunk_tfrecords/chunk_slot_types.pkl \
  --action-mapping $DATA_DIR/finetuning_tfrecords/chunk_tfrecords/chunk_action.pkl \
  --step1-action $STEP1_PATH/result/devtest_action_embedding.json

python3 run_preprocess.py \
  --split devtest \
  --simmc_json $DATA_DIR/simmc2_dials_dstc10_devtest.json \
  --target-txt $STEP1_PATH/data/finetuning_data/chunk/devtest.txt \
  --step1-best-pred $STEP1_PATH/result/prediction_$SPLIT/epoch_5.0_lr_3e-05_ac_0.7_diam_0.7_fr_sys_0.7_ob_num_1.0_sl_0.7_cx_5e-05_sg_2.1e-11_2021-09-23-05-58-14_pred_result.txt \
  --save-path $DATA_DIR/finetuning_data/chunk \
  --slot-candidate $DATA_DIR/slot_candidate.json \
  --slot-mapping $DATA_DIR/finetuning_tfrecords/chunk_tfrecords/chunk_slot_types.pkl \
  --action-mapping $DATA_DIR/finetuning_tfrecords/chunk_tfrecords/chunk_action.pkl \
  --step1-action $STEP1_PATH/result/devtest_action_embedding.json
