from sklearn import metrics
import pickle
import random
import sys
import numpy as np
import torch


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

newresult = []
filename = sys.argv[1]
imagenames = []
with open(sys.argv[2],'r') as tf:
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
    
#with open(sys.argv[2],'rb') as f:
#       pickle.dump(newresult,f)
     furniture =  ['Bed', 'CoffeeTable', 'Sofa', 'Table', 'Shelves', 'CouchChair', 'Chair', 'EndTable', 'Lamp', 'AreaRug']
     fashion = ['vest', 'jeans', 'shirt', 'jacket', 'tshirt', 'trousers', 'hoodie', 'dress', 'suit', 'coat', 'blouse', 'shoes', 'shirt__vest', 'sweater', 'tank_top', 'hat', 'joggers']
     newlogits = []
     i,pred,label,logits = map(list,zip(*results))
     #for logit in logits:
     #    newlogits.append(logit[0].numpy())
     #print(newlogits[0])
#     newpred = np.argmax(np.array(newlogits), axis=2)
     #newpred = np.argmax(softmax(np.array(logits)), axis=-1)
     print(pred)
     with open(sys.argv[3],'w') as tfw:
        for i in range(len(imagenames)):
              tfw.writelines(imagenames[i]+","+fashion[pred[i]]+"\n")
 #    print(newlabel[0])
