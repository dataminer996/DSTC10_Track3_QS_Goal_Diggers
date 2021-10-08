#!/bin/bash

DATA_DIR='../data'
MODEL_DIR='../model'
TRAINDATA_DIR=$DATA_DIR'/finetuning_tfrecords/chunk_tfrecords'
python3 run_finetuning.py \
 --data-dir $DATA_DIR \
 --model-name electra_large \
 --todo-task feature \
 --model-dir $MODEL_DIR \
 --do-predict-split teststd \
 --use-sgnet False \
 --hparams '''{"model_size": "large", "use_tpu": false, "do_train": true, "do_eval": true, "do_predict": true, "task_names": ["chunk"], 
               "traindata_dir": "'${TRAINDATA_DIR}'", 
               "init_checkpoint": "'${MODEL_DIR}'/electra_large"}'''
