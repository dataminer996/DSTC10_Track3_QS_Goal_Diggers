
## TASK 3
Directory Structure: task3/[code | data | model | result]

*download data and finetune model by:*
https://drive.google.com/drive/folders/1ILTFnaRTTcGWAzYXJnt_3QmeiVgE501T?usp=sharing

***
### preprocess
*preprocess dataset for step3 model*
***

cd ./task3/code

`sh run_step3_preprocess.sh`

shell params:
*DATA_DIR: data dir'
*STEP1_PATH: the path of step1*

The shell script above runs the following:
```
python run_preprocess.py \
  --simmc_train_json: official train dataset \
  --simmc_dev_json: official dev dataset \
  --simmc_devtest_json: official devtest dataset \
  --train-target-txt: target train data of step3 \
  --dev-target-txt: target dev data of step3 \
  --devtest-target-txt: target devtest data of step3 \
  --step1-best-pred: best prediction result of step1 \
  --save-path: path to save result \
  --slot-candidate: path of slot candidate \
  --slot-mapping: slot key mapping file \
  --action-mapping: action mapping file \
  --step1-action: action prediction from step1
  ```

***

### feature
*generate feature for step3 model*
***
`run_step3_feature.sh`

shell params:
*DATA_DIR: data dir'
*MODEL_DIR: dir to save model checkpoint'
*TRAINDATA_DIR: dir of tfrecord*

The shell script above runs the following:
```
python run_finetuning.py \
--data-dir: dir of data \
--model-name: electra_large \
--todo-task: feature \
--model-dir:  dir to save model checkpoint \
--do-predict-split: devtest or teststd \
--use-sgnet: False \
--hparams: hparams for electra
```
***

### finetune
*finetune slot value*

*best checkpoint:
for task 1: gs://tangliang-5/track3/step1_finetuning_models/epoch_3.0_lr_2.2e-05_ac_0.7_diam_1.0_fr_sys_0.7_ob_num_0.7_sl_0.7_cx_5e-05_sg_2.1e-11_2021-09-22-08-49-04*
*for task 2/3: gs://tangliang-5/track3/step1_finetuning_models/epoch_5.0_lr_3e-05_ac_0.7_diam_0.7_fr_sys_0.7_ob_num_1.0_sl_0.7_cx_5e-05_sg_2.1e-11_2021-09-22-07-24-53*
***

`run_step3_finetune.sh`

The shell script above runs the following:
```
python run_finetuning.py \
--data-dir: dir of data \
--model-name: electra_large \
--todo-task: finetune \
--model-dir:  dir to save model checkpoint \
--do-predict-split: devtest or teststd \
--use-sgnet: False \
--hparams: hparams of weight, learning rate, epoch
```

***
### predict
*generate final result for task3*
***

`run_step3_predict.sh`

The shell script above runs the following:

```
python format_task3.py \
 --step2-result: path of task2 to get objects id to modify slot pairs \
 --step3-pred-txt: dir of multi-predictions \
 --step3-traget-txt: target file of test phase \
 --slot-mapping-path: path of slot mapping \
 --action-mapping-path: path of action mapping \
 --metadata-path: dir of metadata file \
 --scene-path: dir of image \
 --split-path: official realeased file \
 --save-path: path to save result*
 ```
***


