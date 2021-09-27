#!/bin/bash

SPLIT='devtest'
DATA_DIR='../data'
MODEL_DIR='../model'
TRAINDATA_DIR='../data/finetuning_tfrecords/chunk_tfrecords'
python3 run_finetuning.py \
 --data-dir $DATA_DIR \
 --model-name electra_large \
 --todo-task feature \
 --do-predict-split $SPLIT \
 --use-sgnet True \
 --model-dir $MODEL_DIR \
 --hparams '{"model_size": "large", "use_tpu":false, "task_names": ["chunk"], "traindata_dir": "'${TRAINDATA_DIR}'"}'
