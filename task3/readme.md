## TASK 3
### preprocess

cd ./task3

*preprocess dataset for step3 model (only train, dev)*

sh run_step3_preprocess.sh

### feature
*generate feature for step3 model*

run_step3_feature.sh

### finetune
*finetune model*
*gs://tangliang-5/track3/step2_finetuning_models/epoch_1.0_lr_3e-05_label_1.0_cx_5e-05_sg_2.1e-11_2021-09-22-07-05-50*

run_step1_finetune.sh


### predict
*generate final result for task3*
run_step3_predict.sh

