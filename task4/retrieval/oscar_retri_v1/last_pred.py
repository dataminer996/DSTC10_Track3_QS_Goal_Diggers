import os
import sys
sys.path.append(os.getcwd())

def autotrain(logfilepath,lr,batch_size,weight,epoch,modeldir):

   cmdline = "python3 oscar/run_track3retrievalv1.py  --model_name_or_path pretrained/pretrained_base  --do_eval  --evaluate_during_training  --data_dir data --learning_rate  "  + str(lr)  + "  --output_dir " + str(logfilepath) + "  --gradient_accumulation_steps "  + str(batch_size)  + "  --classoneweight " + str(weight) + " --do_lower_case  --num_train_epochs "  + str(epoch) + " --eval_model_dir  " +  str(modeldir)

   print(cmdline)
   os.system(cmdline)
#   cmdline = "python3 eval_new.py  " + logfilepath
#   print(cmdline)
#   os.system(cmdline)

alldir = sys.argv[1]
allfiles = os.listdir(alldir)
modeldirs = []
for filename in allfiles:
    ret = filename.find(sys.argv[2])
     
    if ret >=0 :
         checkpointdirs = os.listdir(alldir + "/" + filename)
         for checkpointdir in checkpointdirs:
              ret1 = checkpointdir.find("checkpoint")
              if ret1 >= 0:
                 pytorchfiles =  os.listdir(alldir +  "/" + filename + "/" + checkpointdir)  
                 for pytorchfile in pytorchfiles:
                       if pytorchfile == 'pytorch_model.bin':
                          modeldirs.append( alldir + "/" + filename + "/" + checkpointdir) 
               
lr_list = []
logfilepath = 'test'
for modeldir in modeldirs:
     autotrain(logfilepath,5e-5,1,3,1,modeldir)
