import os
import sys

shotnum = sys.argv[1]
waynum = sys.argv[2]
lr = sys.argv[3]
splitdir = sys.argv[4]
outputdir = sys.argv[5]
epoch = sys.argv[6]
cmdline = "python3 train_protonet_load.py --lr " +str(lr) +"  --split_path  " + str(splitdir) + " --temperature 128  \
--max_epoch " + str(epoch) + " --model_type AmdimNet --dataset MiniImageNet \
--model_path max_acc.pth --save_path  "   + " ./" + outputdir + "/track3_lr" + str(lr) + "_" + str(shotnum) + "_" + str(waynum) + "way" +  "_maxepoch" + str(epoch) + "  --shot " + str(shotnum) + "  --way " + str(waynum) + " --query 15 --gpu 1 --step_size 10 --gamma 0.5 --ndf 192 --rkhs 1536 --nd 8 "
print(cmdline)
os.system("nvidia-smi")
os.system(cmdline)
