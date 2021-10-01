from sklearn import metrics
import pickle
import random
import sys
import numpy as np
import torch
import os

def read_train(split_dir):
        csv_path = os.path.join(split_dir, "train_all" + '.csv')
        print(csv_path)

        lines = [x.strip() for x in open(csv_path, 'r',encoding='utf-8').readlines()]

        trainlabel = []
        trainlb_all = 0
        for l in lines:
            name, wnid = l.split(',')
#            if wnid not in trainlabel:
            trainlabel.append(wnid)
        trainlabel_set = list(set(trainlabel))

        trainlb_all = len(trainlabel_set)
        print(trainlabel_set,trainlb_all)
        return trainlabel

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

newresult = []
filename = sys.argv[1]
imagenames = []
valcsv = sys.argv[2] + "/val.csv"
with open(valcsv,'r') as tf:
     lines = tf.readlines()
     for line in lines:
       imagename,_ = line.split(',') 
       imagenames.append(imagename)
 
print(len(lines))    
with open(filename,'rb') as f:
     results = pickle.load(f)
     print(results[1])
     print(len(results))
  
   #  for i,pred,label,logit in result:
   #      pred = pred
   #      label = label.cpu()
   #      logit = logit.cpu()
   #      newresult.appand((i,pred,label,logit))
     furniture = read_train(sys.argv[2])
#with open(sys.argv[2],'rb') as f:
#       pickle.dump(newresult,f)
#     furniture =  ['Bed', 'CoffeeTable', 'Sofa', 'Table', 'Shelves', 'CouchChair', 'Chair', 'EndTable', 'Lamp', 'AreaRug']
     #furniture =  ['Sofa', 'AreaRug', 'CouchChair', 'Bed', 'CoffeeTable', 'Lamp', 'Table', 'EndTable', 'Chair', 'Shelves']
     newlogits = []
     i,pred,label,logits = map(list,zip(*results))
     #for logit in logits:
     #    newlogits.append(logit[0].numpy())
     #print(newlogits[0])
#     newpred = np.argmax(np.array(newlogits), axis=2)
     #newpred = np.argmax(softmax(np.array(logits)), axis=-1)
#     print(pred)
     with open(sys.argv[3],'w') as tfw:
        for i in range(len(imagenames)):
              tfw.writelines(imagenames[i]+","+furniture[pred[i]]+"\n")
 #    print(newlabel[0])
