import os
import sys
sys.path.append(os.getcwd())

def autotrain(logfilepath,lr,batch_size,weight,epoch):
   #cmdline = "python3 oscar/run_objectid.py --learning_rate 1e-6 --model_name_or_path pretrained/pretrained_base/checkpoint-2000000  --do_train --do_eval  --evaluate_during_training  --data_dir data --learning_rate  "  + str(lr)  + "  --output_dir " + str(logfilepath) + "  --gradient_accumulation_steps "  + str(batch_size)  + "  --classoneweight " + str(weight)
   cmdline = "python3 oscar/run_track3retrievalv1.py  --model_name_or_path pretrained/pretrained_base  --do_train --do_eval  --evaluate_during_training  --data_dir data --learning_rate  "  + str(lr)  + "  --output_dir " + str(logfilepath) + "  --gradient_accumulation_steps "  + str(batch_size)  + "  --classoneweight " + str(weight) + " --do_lower_case  --num_train_epochs " + epoch 
   print(cmdline)
   os.system(cmdline)
#   cmdline = "python3 eval_new.py  " + logfilepath
#   print(cmdline)
#   os.system(cmdline)

lr_list = []

#for i in range(int(sys.argv[3])):
#      lr = float(sys.argv[1]) + i * float(sys.argv[2]) 
#      lr_list.append(lr)
lr = sys.argv[1]

batch_size = int(18)
epoch = sys.argv[2]
weight = sys.argv[3]

logfilepath = "./modelgood/find_0822_lowercase_lrnew_"  + str(lr) + "batch" + str(batch_size) + "weight" + weight  + "gpu" + str(1) + "_epoch_" + epoch
autotrain(logfilepath,lr,batch_size,weight,epoch)
