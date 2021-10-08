
## TASK 3
The models of task1 & task3 are baesd on electra (https://github.com/google-research/electra), with AAnet. For task1, we constructed a multi-task model to predict action, disamboguate label, slot key, objects number, from system labels, while task3 we constructed a binary classification model to predict the slot value.

Directory Structure: task3/[code | data | model | result]

*download data and finetune model by:*
https://drive.google.com/drive/folders/1ILTFnaRTTcGWAzYXJnt_3QmeiVgE501T?usp=sharing
gs://tangliang-commit/public/task3
***
### preprocess
*preprocess dataset for step3 model*
***

cd ./task3/code

`sh run_step3_preprocess.sh`
***
####TEST PHASE
`sh run_step3_preprocess_teststd.sh`
***

shell params:
*DATA_DIR: data dir
*STEP1_PATH: the path of step1*

The shell script above runs the following:
```
python run_preprocess.py \
  --split: train,dev,devtest,teststd \
  --simmc_json: official dataset \
  --target-txt: target data of step3 \
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
***
####TEST PHASE
`sh run_step3_feature_teststd.sh`
***

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

*embedding checkpoint:
gs://tangliang-commit/public/task3/model/epoch_1.0_lr_3e-05_label_1.0_cx_0.01_sg_0.0
gs://tangliang-commit/public/task3/model/epoch_1.0_lr_3e-05_label_1.0_cx_0.01_sg_0.0_2021_9_26
***

`run_step3_finetune.sh`
***
####TEST PHASE
`sh run_step3_finetune_teststd.sh`
***

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
***
####TEST PHASE
`sh run_step3_predict_teststd.sh`
***


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


