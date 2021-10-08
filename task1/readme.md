
## TASK 1
### preprocess
*preprocess dataset for step1 model*
Directory Structure: task1/[code | data | model | result]

*download data and finetune model by:*
https://drive.google.com/drive/folders/1ILTFnaRTTcGWAzYXJnt_3QmeiVgE501T?usp=sharing
***

cd ./task1/code

`sh run_step1_preprocess.sh`
***
####TEST PHASE
`sh run_step1_preprocess_teststd.sh`
***
shell params:
*DATA_PATH: path of data with from_system (whether use system objects of  previous turn), objects_num (number of objects) label
*STEP1_SAVE_PATH: save path of task1
*SPLIT: devtest or teststd

The shell script above runs the following:
```
python run_preprocess.py \
--simmc_json: path of dialogue \
--split: train/dev/devtest/teststd \
--action-save-path: save path of task1 \
```

***

### feature
*generate feature for step1 model*
***
`run_step1_feature.sh`
***
####TEST PHASE
`sh run_step1_feature_teststd.sh`
***

shell params:
*SPLIT: devtest or teststd

The shell script above runs the following:
```
python run_finetuning.py \
 --data-dir: official released dataset \
 --model-name: electra_large  \
 --todo-task: feature  \
 --do-predict-split: devtest or teststd \
 --use-sgnet: whether use sgnet  \
 --model-dir: init checkpoint of electra \ 
 --hparams
 ```
***

### finetune
*finetune multi-tasks model: disambiguate_label, action, slot key, from_system, objects_num*

*embedding checkpoint: 
gs://tangliang-commit/public/task1/model/epoch_3.0_lr_2.2e-05_ac_0.7_diam_1.0_fr_sys_0.7_ob_num_0.7_sl_0.7_cx_5e-05_sg_2.1e-11_2021-09-22-08-49-04
gs://tangliang-commit/public/task1/model/lr2e-05epoch_5.0_lr_2e-05_ac_0.7_diam_1.0_fr_sys_0.7_ob_num_0.7_sl_0.7_cx_5e-05_sg_2.1e-11_aa_0.1_2021-09-30-03-02-13
gs://tangliang-commit/public/task1/model/lr2.2e-05epoch_3.0_lr_2.2e-05_ac_0.7_diam_1.0_fr_sys_0.7_ob_num_0.7_sl_0.7_cx_5e-05_sg_2.1e-11_aa_0.0_2021-09-27-06-41-23
gs://tangliang-commit/public/task1/model/lr3e-05epoch_5.0_lr_3e-05_ac_0.7_diam_0.7_fr_sys_0.7_ob_num_1.0_sl_0.7_cx_5e-05_sg_2.1e-11_aa_0.1_2021-09-30-07-39-54
gs://tangliang-commit/public/task1/model/epoch_3.0_lr_4e-05_ac_1.0_diam_0.6_fr_sys_0.6_ob_num_0.6_sl_0.6_cx_5e-05_sg_2.1e-11_2021-09-22-07-24-16
gs://tangliang-commit/public/task1/model/epoch_5.0_lr_3e-05_ac_0.7_diam_0.7_fr_sys_0.7_ob_num_1.0_sl_0.7_cx_5e-05_sg_2.1e-11_2021-09-22-07-24-53
gs://tangliang-commit/public/task1/model/lr3e-05epoch_5.0_lr_3e-05_ac_0.7_diam_0.7_fr_sys_0.7_ob_num_1.0_sl_0.7_cx_5e-05_sg_2.1e-11_aa_0.1_2021-09-30-08-38-05
gs://tangliang-commit/public/task1/model/epoch_5.0_lr_3e-05_ac_0.7_diam_0.7_fr_sys_0.7_ob_num_1.0_sl_0.7_cx_5e-05_sg_2.1e-11_2021-09-23-04-56-29
gs://tangliang-commit/public/task1/model/epoch_5.0_lr_3e-05_ac_0.7_diam_0.7_fr_sys_0.7_ob_num_1.0_sl_0.7_cx_5e-05_sg_2.1e-11_2021-09-23-05-58-14
gs://tangliang-commit/public/task1/model/lr3e-05epoch_5.0_lr_3e-05_ac_0.7_diam_0.7_fr_sys_0.7_ob_num_1.0_sl_0.7_cx_5e-05_sg_2.1e-11_aa_0.1_2021-09-30-09-36-54
***

`run_step1_finetune.sh`
***
####TEST PHASE
`sh run_step1_finetune_teststd.sh`
***

The shell script above runs the following:

```
python run_finetuning.py
 --data-dir: data dir of dataset \
 --model-name: electra_large  \
 --todo-task: finetune \ 
 --do-predict-split: devtest or teststd \
 --use-sgnet: True \
 --model-dir: init checkpoint of electra \ 
 --hparams  params of weight, learning rate, epoch
 ```

***
### predict
*generate final result for task1*
***
run_step1_predict.sh
***
####TEST PHASE
`sh run_step1_predict_teststd.sh`
***

The shell script above runs the following:
```
python format_task1.py \
 --step1-pred-txt: prediction dir \
 --split-path: path of dialogue \
 --save-path: path to save result
 ```
***
####from_system label:
{split}_sobject.json contains from_system label, which means:
Determine whether the user_objectid of this case is included in the sys_obj_id of the previous case in this round. If it meets the requirements, mark from_system as 1, otherwise it is 0. If from_system is 1, put the system_obj_id that appeared before and pass it in s_object after deduplication.
***
