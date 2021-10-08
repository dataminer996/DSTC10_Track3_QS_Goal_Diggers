#!/bin/bash


SPLIT='devtest'
DATA_DIR='../data'
# MODEL_DIR='../model'
#MODEL_DIR='gs://tangliang-5/track3/step1_finetuning_models/epoch_3.0_lr_4e-05_ac_1.0_diam_0.6_fr_sys_0.6_ob_num_0.6_sl_0.6_cx_5e-05_sg_2.1e-11_2021-09-22-07-24-16'
SAVE_DIR='../result/prediction_'$SPLIT
#TRAINDATA_DIR='../data/finetuning_tfrecords/chunk_tfrecords'
TRAINDATA_DIR='gs://tangliang-5/task1/chunk_tfrecords'

for i in {'gs://tangliang-5/track3/step1_finetuning_models/epoch_3.0_lr_2.2e-05_ac_0.7_diam_1.0_fr_sys_0.7_ob_num_0.7_sl_0.7_cx_5e-05_sg_2.1e-11_2021-09-22-08-49-04','gs://tangliang-5/track3/step1_finetuning_models/lr2e-05epoch_5.0_lr_2e-05_ac_0.7_diam_1.0_fr_sys_0.7_ob_num_0.7_sl_0.7_cx_5e-05_sg_2.1e-11_aa_0.1_2021-09-30-03-02-13','gs://tangliang-5/track3/step1_finetuning_models/lr2.2e-05epoch_3.0_lr_2.2e-05_ac_0.7_diam_1.0_fr_sys_0.7_ob_num_0.7_sl_0.7_cx_5e-05_sg_2.1e-11_aa_0.0_2021-09-27-06-41-23','gs://tangliang-5/track3/step1_finetuning_models/lr3e-05epoch_5.0_lr_3e-05_ac_0.7_diam_0.7_fr_sys_0.7_ob_num_1.0_sl_0.7_cx_5e-05_sg_2.1e-11_aa_0.1_2021-09-30-07-39-54','gs://tangliang-5/track3/step1_finetuning_models/epoch_3.0_lr_4e-05_ac_1.0_diam_0.6_fr_sys_0.6_ob_num_0.6_sl_0.6_cx_5e-05_sg_2.1e-11_2021-09-22-07-24-16','gs://tangliang-5/track3/step1_finetuning_models/epoch_5.0_lr_3e-05_ac_0.7_diam_0.7_fr_sys_0.7_ob_num_1.0_sl_0.7_cx_5e-05_sg_2.1e-11_2021-09-22-07-24-53','gs://tangliang-5/track3/step1_finetuning_models/lr3e-05epoch_5.0_lr_3e-05_ac_0.7_diam_0.7_fr_sys_0.7_ob_num_1.0_sl_0.7_cx_5e-05_sg_2.1e-11_aa_0.1_2021-09-30-08-38-05','gs://tangliang-5/track3/step1_finetuning_models/epoch_5.0_lr_3e-05_ac_0.7_diam_0.7_fr_sys_0.7_ob_num_1.0_sl_0.7_cx_5e-05_sg_2.1e-11_2021-09-23-04-56-29','gs://tangliang-5/track3/step1_finetuning_models/epoch_5.0_lr_3e-05_ac_0.7_diam_0.7_fr_sys_0.7_ob_num_1.0_sl_0.7_cx_5e-05_sg_2.1e-11_2021-09-23-05-58-14','gs://tangliang-5/track3/step1_finetuning_models/lr3e-05epoch_5.0_lr_3e-05_ac_0.7_diam_0.7_fr_sys_0.7_ob_num_1.0_sl_0.7_cx_5e-05_sg_2.1e-11_aa_0.1_2021-09-30-09-36-54'}
    do
    MODEL_DIR=$i
    echo $MODEL_DIR
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
    done
