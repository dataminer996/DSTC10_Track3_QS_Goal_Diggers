#!/bin/bash


SPLIT='devtest'
DATA_DIR='../data'
# MODEL_DIR='../model'
MODEL_DIR='gs://tangliang-5/track3/step1_finetuning_models/epoch_3.0_lr_4e-05_ac_1.0_diam_0.6_fr_sys_0.6_ob_num_0.6_sl_0.6_cx_5e-05_sg_2.1e-11_2021-09-22-07-24-16'
SAVE_DIR='../result/prediction'
TRAINDATA_DIR='../data/finetuning_tfrecords/chunk_tfrecords'
python3 run_finetuning.py  \
 --data-dir $DATA_DIR  \
 --model-name electra_large \
 --todo-task finetune \
 --model-dir $MODEL_DIR \
 --do-predict-split $SPLIT \
 --use-sgnet True \
 --hparams '''{"model_size": "large", "use_tpu": true, "do_predict": true, "task_names": ["chunk"], 
             "traindata_dir": "'${TRAINDATA_DIR}'", "save_prediction": "'${SAVE_DIR}'", 
             "init_checkpoint": "gs://tangliang-5/track3/electra_large", "learning_rate": 4e-05, "num_train_epochs": 3.0, "action_weight": 1.0, 
             "disambiguate_weight": 0.6, "from_system_weight": 0.6, "slot_weight": 0.6, "cx_weight": 5e-05, "sg_weight": 2.1e-11, "objects_num_weight": 0.6}'''
             
MODEL_DIR='gs://tangliang-5/track3/step1_finetuning_models/epoch_5.0_lr_3e-05_ac_0.7_diam_0.7_fr_sys_0.7_ob_num_1.0_sl_0.7_cx_5e-05_sg_2.1e-11_2021-09-22-07-24-53'
python3 run_finetuning.py  \
 --data-dir $DATA_DIR  \
 --model-name electra_large \
 --todo-task finetune \
 --model-dir $MODEL_DIR \
 --do-predict-split $SPLIT \
 --use-sgnet True \
 --hparams '''{"model_size": "large", "use_tpu": true, "do_predict": true, "task_names": ["chunk"], 
             "traindata_dir": "'${TRAINDATA_DIR}'", "save_prediction": "'${SAVE_DIR}'", 
             "init_checkpoint": "gs://tangliang-5/track3/electra_large", "learning_rate": 3e-05, "num_train_epochs": 5.0, "action_weight": 0.7, 
             "disambiguate_weight": 0.7, "from_system_weight": 0.7, "slot_weight": 0.7, "cx_weight": 5e-05, "sg_weight": 2.1e-11, "objects_num_weight": 1.0}'''

MODEL_DIR='gs://tangliang-5/track3/step1_finetuning_models/epoch_3.0_lr_2.2e-05_ac_0.7_diam_1.0_fr_sys_0.7_ob_num_0.7_sl_0.7_cx_5e-05_sg_2.1e-11_2021-09-22-08-49-04'
python3 run_finetuning.py  \
 --data-dir $DATA_DIR  \
 --model-name electra_large \
 --todo-task finetune \
 --model-dir $MODEL_DIR \
 --do-predict-split $SPLIT \
 --use-sgnet True \
 --hparams '''{"model_size": "large", "use_tpu": true, "do_predict": true, "task_names": ["chunk"], 
             "traindata_dir": "'${TRAINDATA_DIR}'", "save_prediction": "'${SAVE_DIR}'", 
             "init_checkpoint": "gs://tangliang-5/track3/electra_large", "learning_rate": 2.2e-05, "num_train_epochs": 3.0, "action_weight": 0.7, 
             "disambiguate_weight": 1.0, "from_system_weight": 0.7, "slot_weight": 0.7, "cx_weight": 5e-05, "sg_weight": 2.1e-11, "objects_num_weight": 0.7}'''