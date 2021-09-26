import os
import sys
sys.path.append(os.getcwd())

def autotrain(logfilepath,lr,batch_size,epoch):
   cmdline = "python3 oscar/run_genanswer.py  --model_name_or_path pretrained/pretrained_base  --do_train --do_eval  --evaluate_during_training  --data_dir data --learning_rate  "  + str(lr)  + "  --output_dir " + str(logfilepath) + "  --gradient_accumulation_steps "  + str(batch_size)  + "  --num_train_epochs " + epoch + "      --do_lower_case    --add_od_labels      --tie_weights    --freeze_embedding      --label_smoothing 0.1   --drop_worst_ratio 0.2  --drop_worst_after 20000 "
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
gpu = 1
#for i in lr_list:
outputfilepath = "./modelgood/find_0822lrnew_"  + str(lr) + "batch" + str(batch_size)  + "gpu" + str(gpu)  + "_epoch_" + epoch
autotrain(outputfilepath,lr,batch_size,epoch)
