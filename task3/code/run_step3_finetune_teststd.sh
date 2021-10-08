#!/bin/bash

DATA_DIR='gs://tangliang-5/task3'
# MODEL_DIR='../model'
TRAINDATA_DIR=$DATA_DIR'/finetuning_tfrecords/chunk_tfrecords'
SPLIT='teststd'
SAVE_PATH='../result/prediction_'$SPLIT

for i in {'gs://tangliang-5/track3/step2_finetuning_models/epoch_1.0_lr_3e-05_label_1.0_cx_0.01_sg_0.0','gs://tangliang-5/track3/step2_finetuning_models/epoch_1.0_lr_3e-05_label_1.0_cx_0.01_sg_0.0_2021_9_26'}
    do
    MODEL_DIR=$i
    echo $MODEL_DIR
    python3 run_finetuning.py \
     --data-dir $DATA_DIR \
     --model-name electra_large \
     --todo-task finetune \
     --model-dir $MODEL_DIR \
     --do-predict-split $SPLIT \
     --use-sgnet False \
     --hparams '''{"model_size": "large", "use_tpu": true, "do_eval": false, "do_predict": true, "task_names": ["chunk"], 
                   "traindata_dir": "'${TRAINDATA_DIR}'", "save_prediction": "'${SAVE_PATH}'", \
                   "init_checkpoint": "gs://tangliang-5/track3/electra_large", "learning_rate": 3e-05, "num_train_epochs": 1.0, 
                   "label_weight": 1.0, "cx_weight": 5e-05, "sg_weight": 0.0}'''
    done
