import pickle
import random
import sys
import numpy as np
filename = sys.argv[1]
with open(filename,'rb') as f:
     features = pickle.load(f)
new_features = []

neg = 0
pos = 1
for feature in features:
      dialogue_final,scene_thisround,did,id_final,type_final,bbox_final,label_final,image_feature = feature
      if label_final == 0:
            neg = neg + 1
      else:
            pos = pos +1
print(pos,neg)
#outfilename = sys.argv[2]        
#print("done features",len(new_features))    
#with open(outfilename,'wb') as fp:
#         pickle.dump(new_features,fp) 
                            

