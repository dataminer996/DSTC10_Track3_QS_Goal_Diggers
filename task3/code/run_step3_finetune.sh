#!/bin/bash

DATA_DIR='../data'
# MODEL_DIR='../model'
MODEL_DIR='gs://tangliang-5/track3/step2_finetuning_models/epoch_1.0_lr_3e-05_label_1.0_cx_5e-05_sg_2.1e-11_2021-09-22-07-05-50'
TRAINDATA_DIR=$DATA_DIR'/finetuning_tfrecords/chunk_tfrecords'
python3 run_finetuning.py \
 --data-dir $DATA_DIR \
 --model-name electra_large \
 --todo-task finetune \
 --model-dir $MODEL_DIR \
 --do-predict-split devtest \
 --use-sgnet False \
 --hparams '''{"model_size": "large", "use_tpu": true, "do_eval": false, "do_predict": true, "task_names": ["chunk"], 
               "traindata_dir": "'${TRAINDATA_DIR}'", 
               "init_checkpoint": "gs://tangliang-5/track3/electra_large", "learning_rate": 3e-05, "num_train_epochs": 1.0, 
               "label_weight": 1.0, "cx_weight": 5e-05, "sg_weight": 0.0}'''
