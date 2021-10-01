from sklearn import metrics
import pickle
import random
import sys
import numpy as np
import torch
import os
from scipy.special import softmax
np.set_printoptions(threshold=100000)

def read_train(split_dir):
        csv_path = os.path.join(split_dir, "train" + '.csv')
        print(csv_path)

        lines = [x.strip() for x in open(csv_path, 'r',encoding='utf-8').readlines()]

        trainlabel = []
        trainlb_all = 0
        for l in lines:
            name, wnid = l.split(',')
            if wnid not in trainlabel:
               trainlb_all = trainlb_all + 1
               trainlabel.append(wnid)
#        trainlabel_set = list(set(trainlabel))

 #       trainlb_all = len(trainlabel_set)
        print(trainlabel,trainlb_all)
        return trainlabel

#def softmax(x):
#    return np.exp(x)/np.sum(np.exp(x), axis=-1)

imagenames = []
newresult = []
valcsv = sys.argv[1] + "/val.csv"
with open(valcsv,'r') as tf:
     lines = tf.readlines()
     for line in lines:
       imagename,_ = line.split(',') 
       imagenames.append(imagename)
 
pre_types  = sys.argv[1] + "/trainlabelset.bin"
with open(pre_types,'rb') as f:
     pred_type_list  = pickle.load(f)
     print(pred_type_list)


def one_hot(index,num):
    label = []
    for i in range(num):
        if i == index:
           label.append(1)
        else:
           label.append(0)
    return np.array(label)


def checkmax(c):
    count = 0
    max_num = np.max(c)
    for num in c:
        if num == max_num:
            count = count +1
    return count

print(len(lines))    
def readtheresult(filename):
  print(filename)
  with open(filename,'rb') as f:
     results = pickle.load(f)
     print(results[1])
     print(len(results))
  
     newlogits = []
     i,pred,label,logits = map(list,zip(*results))
     for logit in logits:
         newlogits.append(logit.numpy())
     print(newlogits[0])
    # newpred = np.argmax(np.array(newlogits), axis=2)
     prob = softmax(np.array(newlogits),axis=-1)
     print(prob[0])
      
     newpred = np.argmax(softmax(np.array(newlogits),axis=-1), axis=-1)
     print("pred10",pred[0:10])
     print("newpred10",newpred[0:10])
     return np.array(prob),np.array(pred)

alldir = sys.argv[2]
modeldirs = os.listdir(sys.argv[2])
resultfiles = []
for filename in modeldirs:
         checkpointdirs = os.listdir(alldir +  "/" + filename +"/MiniImageNet-AmdimNet-ProtoNet"   )
         for checkpointdir in checkpointdirs:
              checkdir = alldir +  "/" + filename +"/MiniImageNet-AmdimNet-ProtoNet/" + checkpointdir
              if os.path.exists(checkdir + "/"  + "result.bin"):
                               resultfiles.append(checkdir+ "/"  + "result.bin")


probs = []
preds = []
for  resultfile  in  resultfiles:
     #print(resultfile)
     prob,pred = readtheresult(resultfile)
     prob = prob.reshape((len(pred),len(pred_type_list)))
     probs.append(prob)
     preds.append(pred)

results = []
print(preds[0].shape)
print(probs[0].shape)
for i in range(len(preds[0])):
     pred_num = None
     for j in range(len(preds)):
        if pred_num is None:
          pred_num = one_hot(preds[j][i],len(pred_type_list))        
        else:
          pred_num = pred_num + one_hot(preds[j][i],len(pred_type_list))        
     if checkmax(pred_num) == 1:
          results.append(np.argmax(pred_num))
          continue
     prob_total = None
     for j in range(len(probs)):
        if prob_total is None:
           prob_total = np.array(probs[j][i])
        else:
           prob_total = np.array(probs[j][i]) + prob_total

     results.append(np.argmax(prob_total))
        


print("results",results[0:10])

with open(sys.argv[3],'w') as tfw:
        for i in range(len(imagenames)):
              tfw.writelines(imagenames[i]+","+pred_type_list[results[i]]+"\n")
 #    print(newlabel[0])
