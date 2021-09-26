import json
import jsonpath
import torch
import numpy as np
import pickle
import os, sys
import argparse
import copy
import calculate_bleu

# 生成所有devtest中的对应的objectid集合
dir_json = 'data/simmc2_dials_dstc10_devtest.json'
dir_bin = 'D:/data/simmc2/result.bin'

# 总函数

alldir = sys.argv[2]
allfiles = os.listdir(alldir)
modeldirs = []
for filename in allfiles:
    ret = filename.find(sys.argv[1])
    if ret >=0 :
         checkpointdirs = os.listdir( alldir + "/" + filename)
         for checkpointdir in checkpointdirs:
              ret1 = checkpointdir.find("checkpoint")
              if ret1 >= 0:
                 pytorchfiles =  os.listdir(alldir + "/" + filename + "/" + checkpointdir)
                 for pytorchfile in pytorchfiles:
                       if pytorchfile == 'pytorch_model.bin':
                          if os.path.exists(alldir + "/" + filename + "/" + checkpointdir + "/resulefile.txt"):
                               modeldirs.append( alldir + "/" + filename + "/" + checkpointdir+ '/resulefile.txt')
#                          if os.path.exists("./" + filename + "/" + checkpointdir + "/resultfromsys.txt"):
#                               modeldirs.append( "./" + filename + "/" + checkpointdir + "/resultfromsys.txt")
               
for modeldir in modeldirs:
       #print(modeldir)
       with open(modeldir,'r') as tf:
             lines = tf.readlines()
             #print(len(lines))
             if len(lines) > 8000:
                 score,_ = calculate_bleu.calculate_bleu("./data/simmc2_dials_dstc10_devtest.json",modeldir)
                 print(modeldir,"\t",score)
                 #cmdline = "python3 calculate_bleu.py ./data/simmc2_dials_dstc10_devtest.json  " +  modeldir
                 #print(cmdline)
                 #os.system(cmdline) 

