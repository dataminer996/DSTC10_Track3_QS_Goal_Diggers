import os
import sys
sys.path.append(os.getcwd())

def autotrain(logfilepath,lr,batch_size,weight,epoch,modeldir):
   #cmdline = "python3 oscar/run_objectid.py --learning_rate 1e-6 --model_name_or_path pretrained/pretrained_base/checkpoint-2000000  --do_train --do_eval  --evaluate_during_training  --data_dir data --learning_rate  "  + str(lr)  + "  --output_dir " + str(logfilepath) + "  --gradient_accumulation_steps "  + str(batch_size)  + "  --classoneweight " + str(weight)

   cmdline = "python3 oscar/run_genanswer.py  --model_name_or_path pretrained/pretrained_base  --do_eval  --evaluate_during_training  --data_dir data --learning_rate  "  + str(lr)  + "  --output_dir " + str(logfilepath) + "  --gradient_accumulation_steps "  + str(batch_size)  + "  --num_train_epochs " + str(epoch) + "      --do_lower_case    --add_od_labels      --tie_weights    --freeze_embedding      --label_smoothing 0.1   --drop_worst_ratio 0.2  --drop_worst_after 20000 "  + " --eval_model_dir  " +  str(modeldir) + "   --num_beams 5  --data_dir data  --per_gpu_eval_batch_size 1 " 
 
   #cmdline = "python3 oscar/run_objectid.py  --model_name_or_path pretrained/pretrained_base  --do_eval  --evaluate_during_training  --data_dir data --learning_rate  "  + str(lr)  + "  --output_dir " + str(logfilepath) + "  --gradient_accumulation_steps "  + str(batch_size)  + "  --classoneweight " + str(weight) + "   --num_train_epochs "  + str(epoch) + " --eval_model_dir  " +  str(modeldir)
   print(cmdline)
   os.system(cmdline)
#   cmdline = "python3 eval_new.py  " + logfilepath
#   print(cmdline)
#   os.system(cmdline)

alldir = sys.argv[1]
allfiles = os.listdir(alldir)
modeldirs = []
for filename in allfiles:
    print(filename)
    ret = filename.find(sys.argv[2])
    print(ret)
     
    if ret >=0 :
         checkpointdirs = os.listdir( alldir +"/" + filename)
         for checkpointdir in checkpointdirs:
              ret1 = checkpointdir.find("checkpoint")
              if ret1 >= 0:
                 pytorchfiles =  os.listdir( alldir +"/" + filename + "/" + checkpointdir)  
                 for pytorchfile in pytorchfiles:
                       if pytorchfile == 'pytorch_model.bin':
                        # if os.path.exists(alldir + "/" + filename + "/" + checkpointdir + "/resulefile.txt"):
                        #       continue
                        # else:
                               #modeldirs.append( "./" + filename + "/" + checkpointdir+ '/resulefile.txt')
                               modeldirs.append( alldir +"/" + filename + "/" + checkpointdir) 
                       
print(modeldirs)
for modeldir in modeldirs:
     print(modeldir)
lr_list = []
logfilepath =  'test'
for modeldir in modeldirs:
     print(modeldir)
     autotrain(logfilepath,5e-5,1,3,1,modeldir)
