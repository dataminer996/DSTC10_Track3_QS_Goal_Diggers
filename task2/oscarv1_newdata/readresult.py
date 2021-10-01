import pickle
import random
import sys
import numpy as np
filename = sys.argv[1]
with open(filename,'rb') as f:
     features = pickle.load(f)
new_features = []

for feature in features:
      dids,preds,labels,logitsoutput = feature
      print(dids)
      print(preds)
      print(labels)
      print(logitsoutput) 
#outfilename = sys.argv[2]        
#print("done features",len(new_features))    
#with open(outfilename,'wb') as fp:
#         pickle.dump(new_features,fp) 
                            

