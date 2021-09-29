
## TASK 1
### preprocess
*preprocess dataset for step1 model*
Directory Structure: task1/[code | data | model | result]

*download data and finetune model by:*
https://drive.google.com/drive/folders/1ILTFnaRTTcGWAzYXJnt_3QmeiVgE501T?usp=sharing
***

cd ./task1/code

`sh run_step1_preprocess.sh`

shell params:
*DATA_PATH: path of data with from_system (whether use system objects of  previous turn), objects_num (number of objects) label
*STEP1_SAVE_PATH: save path of task1
*STEP3_SAVE_PATH: save path of task3
*SPLIT: devtest or teststd

The shell script above runs the following:
```
python run_preprocess.py \
--simmc_json: path of dialogue \
--split: train/dev/devtest/teststd \
--action-save-path: save path of task1 \
--slot-save-path: save path of task3 \
--slot-candidate-path: candidates of slot values (generate by `python slot_candidate.py`)
```

***

### feature
*generate feature for step1 model*
***
`run_step1_feature.sh`

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

*best checkpoint:
for task 1: gs://tangliang-5/track3/step1_finetuning_models/epoch_3.0_lr_2.2e-05_ac_0.7_diam_1.0_fr_sys_0.7_ob_num_0.7_sl_0.7_cx_5e-05_sg_2.1e-11_2021-09-22-08-49-04*
*for task 2/3: gs://tangliang-5/track3/step1_finetuning_models/epoch_5.0_lr_3e-05_ac_0.7_diam_0.7_fr_sys_0.7_ob_num_1.0_sl_0.7_cx_5e-05_sg_2.1e-11_2021-09-22-07-24-53*
***

`run_step1_finetune.sh`
need to use TPU
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

The shell script above runs the following:
```
python format_task1.py \
 --step1-pred-txt: prediction dir \
 --split-path: path of dialogue \
 --save-path: path to save result
 ```
***
