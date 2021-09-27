import json
import datetime
import os

# "learning_rate": [2.00E-05, 3.00E-05, 5.00E-05, 8.00E-05, 1.10E-04, 1.50E-04, 2.00E-04, 3.00E-04, 4.00E-04, 5.00E-04],
dict_all = [[2e-05,3.0,1.0,0.7,0.7,0.7,5e-05,2.1e-11,0.7],
[3e-05,3.0,1.0,0.8,0.8,0.8,5e-05,2.1e-11,0.8],
[2.2e-05,3.0,0.7,1.0,0.7,0.7,5e-05,2.1e-11,0.7],
[2.2e-05,3.0,1.0,0.7,0.7,0.7,5e-05,2.1e-11,0.7],
[4e-05,3.0,1.0,0.01,0,0.01,5e-05,2.1e-11,0.01],
[2e-05,3.0,1.0,1.0,0.7,0.7,5e-05,2.1e-11,0.7],
[3e-05,3.0,1.0,0.4,0.4,0.4,5e-05,2.1e-11,0.4],
[3e-05,3.0,1.0,0.7,0.7,0.7,5e-05,2.1e-11,0.7],
[4e-05,3.0,1.0,0.6,0.6,0.6,5e-05,2.1e-11,0.6],
[4e-05,3.0,1.0,0.8,0.8,0.8,5e-05,2.1e-11,0.8],
[2e-05,5.0,0.7,1.0,0.7,0.7,5e-05,2.1e-11,0.7],
[2.2e-05,3.0,0.7,0.7,1.0,0.7,5e-05,2.1e-11,0.7],
[3e-05,5.0,0.7,0.7,0.7,0.7,5e-05,2.1e-11,1.0],
[2e-05,3.0,1.0,0.1,0.1,0.1,5e-05,2.1e-11,0.1],
[1.8e-05,3.0,0.7,1.0,0.7,0.7,5e-05,2.1e-11,0.7],
[2.2e-05,3.0,0.7,0.7,0.7,0.7,5e-05,2.1e-11,1.0]]
# dict_all = {
#     "num_train_epochs": [7.0],
#     "learning_rate": [4e-05],
#     "action_weight": [0.0],
#     "disambiguate_weight": [0.0],
#     "from_system_weight": [1.0],
#     "objects_num_weight": [0.0],
#     "slot_weight": [0.0],
#     "cx_weight": [0.0],
#     "sg_weight": [0.0],
#     "aa_weight": [0.0]
# }
print('===========================')
print('version: 2021-9-22')
print(dict_all)

json_test = {"model_size": "large", "use_tpu": True, "do_predict": False, "task_names": ["chunk"], \
             "traindata_dir": "gs://tangliang-5/track3/electra_step1/data/models/electra_large/finetuning_tfrecords/chunk_tfrecords", \
             "init_checkpoint": "gs://tangliang-5/track3/electra_large", }

# for key, value in dict_all.items():
#     if len(value) == 1:
#         json_test[key] = value[0]

key_list = ['learning_rate','num_train_epochs','action_weight','disambiguate_weight','from_system_weight','slot_weight','cx_weight','sg_weight','objects_num_weight']
for each in dict_all:
    # json_test['learning_rate'] = each[0]
    # print(json_test)
    for i in range(len(key_list)):
        json_test[key_list[i]] = each[i]
    print(json_test)
    modeldir = 'epoch_' + str(each[1]) + '_lr_'+str(each[0]) + '_ac_'+str(each[2])+'_diam_'+str(each[3])\
               +'_fr_sys_'+str(each[4])+'_ob_num_'+str(each[8])+'_sl_'\
               +str(each[5])+'_cx_'+str(each[6])+'_sg_'+str(each[7]) + '_' +  datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d-%H-%M-%S")
    command = '''python run_finetuning.py \
                    --data-dir gs://tangliang-5/track3/electra_step1/data \
                    --model-name electra_large \
                    --todo-task finetune \
                    --model-dir gs://tangliang-5/track3/step1_finetuning_models/{}\
                    --do-predict-split devtest \
                    --use-sgnet true \
                    --hparams '{}' '''.format(modeldir, json.dumps(json_test))
    print('command', command)
    result = os.system(command)
    # command = '''gsutil rm gs://tangliang-5track3/step1_finetuning_models/demo1/events*
    #                 gsutil rm gs://tangliang-5/track3/step1_finetuning_models/demo1/model*
    #                 gsutil rm gs://tangliang-5/track3/step1_finetuning_models/demo1/graph*
    #                 gsutil rm gs://tangliang-5/track3/step1_finetuning_models/demo1/checkpoint*
    #                 gsutil rm gs://tangliang-5/track3/step1_finetuning_models/demo1/eval_results.txt'''
    # print('command', command)
    #result = os.system(command)