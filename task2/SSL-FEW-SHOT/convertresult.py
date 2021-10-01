from sklearn import metrics
import pickle
import random
import sys
import numpy as np
import torch

newresult = []
filename = sys.argv[1]
with open(filename,'rb') as f:
     results = pickle.load(f)
     print(results[1])
  
     for i,pred,label,logit in results:
         pred = pred.cpu()
         label = label.cpu()
         logit = logit.cpu()
         newresult.append((i,pred,label,logit))
    
with open(sys.argv[2],'wb') as f:
       pickle.dump(newresult,f)
    
#     ret = metrics.classification_report(label_index.cpu(),pred.cpu(),target_names=None,digits=4,output_dict=False)
 #    print(ret)
