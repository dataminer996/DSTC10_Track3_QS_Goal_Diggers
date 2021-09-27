import json
import datetime
import os

# "learning_rate": [2.00E-05, 3.00E-05, 5.00E-05, 8.00E-05, 1.10E-04, 1.50E-04, 2.00E-04, 3.00E-04, 4.00E-04, 5.00E-04],
# dict_all = {
#     "num_train_epochs": [2.0],
#     "learning_rate": [3.00E-05, 5.00E-05, 8.00E-05],
#     "label_weight": [1.0],
#     "cx_weight": [0.0],
#     "sg_weight": [0.0]
# }
dict_all = [[3e-05,1.0,1.0,5e-05,2.1e-11],
[3e-05,1.0,1.0,0.01,0.01],
[4e-05,1.0,1.0,5e-05,2.1e-11]]
print('===========================')
print('step2 version: 2021-9-22')
print(dict_all)

json_test = {"model_size": "large", "use_tpu": True, "do_predict": False, "task_names": ["chunk"], \
             "traindata_dir": "gs://tangliang-5/track3/electra_slot/data/models/electra_large/finetuning_tfrecords/chunk_tfrecords", \
             "init_checkpoint": "gs://tangliang-5/track3/electra_large", }

key_list = ['learning_rate','num_train_epochs','label_weight','cx_weight','sg_weight']
for each in dict_all:
    # json_test['learning_rate'] = each[0]
    # print(json_test)
    for i in range(len(key_list)):
        json_test[key_list[i]] = each[i]
    print(json_test)

    modeldir = 'epoch_' + str(each[1]) + '_lr_'+str(each[0]) + '_label_'\
               +str(each[2])+'_cx_'+str(each[3])+'_sg_'+str(each[4]) + '_' +  datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d-%H-%M-%S")
               
    command = '''python run_finetuning.py \
                    --data-dir gs://tangliang-5/track3/electra_slot/data \
                    --model-name electra_large \
                    --todo-task finetune \
                    --model-dir gs://tangliang-5/track3/step2_finetuning_models/{}\
                    --do-predict-split devtest \
                    --use-sgnet true \
                    --hparams '{}' '''.format(modeldir, json.dumps(json_test))
    print('command', command)
    result = os.system(command)
    # command = '''gsutil rm gs://tangliang-5/track3/step1_finetuning_models/demo1/events*
    #                 gsutil rm gs://tangliang-5/track3/step1_finetuning_models/demo1/model*
    #                 gsutil rm gs://tangliang-5/track3/step1_finetuning_models/demo1/graph*
    #                 gsutil rm gs://tangliang-5/track3/step1_finetuning_models/demo1/checkpoint*
    #                 gsutil rm gs://tangliang-5/track3/step1_finetuning_models/demo1/eval_results.txt'''
    # print('command', command)
    #result = os.system(command)