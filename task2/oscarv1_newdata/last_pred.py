import os
import sys
sys.path.append(os.getcwd())

def autotrain(logfilepath,lr,batch_size,weight,epoch,modeldir):

   if modeldir.find('large') >= 0:
   #cmdline = "python3 oscar/run_objectid.py --learning_rate 1e-6 --model_name_or_path pretrained/pretrained_base/checkpoint-2000000  --do_train --do_eval  --evaluate_during_training  --data_dir data --learning_rate  "  + str(lr)  + "  --output_dir " + str(logfilepath) + "  --gradient_accumulation_steps "  + str(batch_size)  + "  --classoneweight " + str(weight)
        cmdline = "python3 oscar/run_objectid.py  --model_name_or_path pretrained/pretrained_large  --do_eval  --evaluate_during_training  --data_dir data --learning_rate  "  + str(lr)  + "  --output_dir " + str(logfilepath) + "  --gradient_accumulation_steps "  + str(batch_size)  + "  --classoneweight " + str(weight) + " --do_lower_case  --num_train_epochs "  + str(epoch) + " --eval_model_dir  " +  str(modeldir) 
   else:
        cmdline = "python3 oscar/run_objectid.py  --model_name_or_path pretrained/pretrained_base  --do_eval  --evaluate_during_training  --data_dir data --learning_rate  "  + str(lr)  + "  --output_dir " + str(logfilepath) + "  --gradient_accumulation_steps "  + str(batch_size)  + "  --classoneweight " + str(weight) + " --do_lower_case  --num_train_epochs "  + str(epoch) + " --eval_model_dir  " +  str(modeldir)

   print(cmdline)
   os.system(cmdline)
#   cmdline = "python3 eval_new.py  " + logfilepath
#   print(cmdline)
#   os.system(cmdline)

alldir = sys.argv[1]
allfiles = os.listdir(alldir)
modeldirs = []
for filename in allfiles:
    if len(sys.argv) >= 3:
        ret = filename.find(sys.argv[2])
    else:
        ret = 1
 #   ret1 = filename.find('lowcase')

    if ret >=0:
         checkpointdirs = os.listdir(alldir + "/" + filename)
         for checkpointdir in checkpointdirs:
              ret1 = checkpointdir.find("checkpoint")
              if ret1 >= 0:
                 pytorchfiles =  os.listdir(alldir +  "/" + filename + "/" + checkpointdir)  
                 for pytorchfile in pytorchfiles:
                       if pytorchfile == 'pytorch_model.bin':
                          modeldirs.append( alldir + "/" + filename + "/" + checkpointdir) 
               
lr_list = []
logfilepath = sys.argv[1] + 'test'
for modeldir in modeldirs:
     autotrain(logfilepath,5e-5,1,3,1,modeldir)
