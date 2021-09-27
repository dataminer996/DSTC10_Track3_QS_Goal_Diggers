
## TASK 1
### preprocess
*preprocess dataset for step1 model*
***

cd ./task1

s_objects_produce.py:
dir_input- dir of train,dev or devtest.json
Obtain the json file according to the input path, traverse each round and each  case in each round. Determine whether the user_objectid of this case is included in the sys_obj_id of the previous use case in this round.(for example, all elements of user_object of the eighth case appear in the system_object of cases 0-7, and no new elements appear) If it meets the requirements, mark from_system as 1, otherwise it is 0. If from_system is 1, put the system_obj_id that appeared before and pass it in s_object after deduplication.

`sh run_step1_preprocess.sh`

*shell params:
DATA_PATH: path of data with from_system (whether use system objects of  previous turn), objects_num (number of objects) label
STEP1_SAVE_PATH: save path of task1
STEP3_SAVE_PATH: save path of task3
SPLIT: devtest or teststd*

*run_preprocess.py params:
simmc_json: path of dialogue
split: train/dev/devtest/teststd
action-save-path: save path of task1
slot-save-path: save path of task3
slot-candidate-path: candidates of slot values (generate by `python slot_candidate.py`)*
***

### feature
*generate feature for step1 model*
***
`run_step1_feature.sh`

*shell params:
SPLIT: devtest or teststd*

*run_finetuning.py params:
 data-dir: data dir of dataset
 model-name: electra_large 
 todo-task: feature or finetune 
 do-predict-split: devtest or teststd
 use-sgnet: whether use sgnet 
 model-dir: init checkpoint of electra 
 hparams*
***

### finetune
*finetune multi-tasks model: disambiguate_label, action, slot key, from_system, objects_num*

*best checkpoint:
for task 1: gs://tangliang-5/track3/step1_finetuning_models/epoch_3.0_lr_2.2e-05_ac_0.7_diam_1.0_fr_sys_0.7_ob_num_0.7_sl_0.7_cx_5e-05_sg_2.1e-11_2021-09-22-08-49-04*
*for task 2/3: gs://tangliang-5/track3/step1_finetuning_models/epoch_5.0_lr_3e-05_ac_0.7_diam_0.7_fr_sys_0.7_ob_num_1.0_sl_0.7_cx_5e-05_sg_2.1e-11_2021-09-22-07-24-53*
***

`run_step1_finetune.sh`

*run_finetuning.py params:
 data-dir: data dir of dataset
 model-name: electra_large 
 todo-task: feature or finetune 
 do-predict-split: devtest or teststd
 use-sgnet: whether use sgnet 
 model-dir: init checkpoint of electra 
 hparams  params of weight, learning rate, epoch*

***
### predict
*generate final result for task1*
***
run_step1_predict.sh

*format_task1.py params:
 step1-pred-txt: prediction dir
 split-path: path of dialogue
 save-path: path to save result*
***
