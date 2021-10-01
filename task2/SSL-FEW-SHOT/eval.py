import os
import sys

shotnum = sys.argv[1]
waynum = sys.argv[2]
modelpath = sys.argv[3]

modelpath = modelpath + "/MiniImageNet-AmdimNet-ProtoNet"
dirnames = os.listdir(modelpath)
print(dirnames)
for dirname in dirnames:
   modeldir = os.path.join(modelpath,dirname)
   print(modeldir)
   filenames = os.listdir(modeldir)
   print(filenames)
   if "max_acc.pth" in filenames:
        modelname = os.path.join(modelpath,dirname,"max_acc.pth")
           

cmdline = "python3 eval_protonet.py  --model_type AmdimNet   --dataset MiniImageNet    --model_path  " + modelname + "  --shot " + str(shotnum) + "  --way " + str(waynum) + " --query 1 --gpu 1  --ndf 192 --rkhs 1536 --nd 8 "
print(cmdline)
os.system("nvidia-smi")
os.system(cmdline)

