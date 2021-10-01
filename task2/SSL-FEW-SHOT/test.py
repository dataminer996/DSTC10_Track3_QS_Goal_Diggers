import os
import sys

shotnum = sys.argv[1]
waynum = sys.argv[2]
modelpath = sys.argv[3]
split_path = sys.argv[4]
image_path = sys.argv[5]

os.system("nvidia-smi")

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
           

cmdline = "python3 test_protonet.py  --model_type AmdimNet --image_path "  + image_path  +    "  --dataset MiniImageNet    --model_path  " + modelname + "  --shot " + str(shotnum) + "  --way " + str(waynum) + " --query 1 --gpu 1  --ndf 192 --rkhs 1536 --nd 8 --split_path " + split_path
print(cmdline)
os.system("nvidia-smi")
os.system(cmdline)


