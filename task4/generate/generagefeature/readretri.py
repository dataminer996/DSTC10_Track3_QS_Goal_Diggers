import pickle
import random
import sys
import numpy as np
filename = sys.argv[1]
num = 0
with open(filename,'rb') as f:
     features = pickle.load(f)
     for feature in features:
         dialogue_finals, scene_thisrounds,dialog_ids, id_finals, type_finals, bbox_finals, slotvalue_finals, image_features,label,index = feature 
        # if label == 0:
         print(dialog_ids,dialogue_finals,index)
         num = num + 1
         if num > 3:
                break
 
print(len(features))
#print(features[0]) 
#print(features[38072]) 
#print(features[76142]) 
#print(features[114214]) 
#print(features[-1]) 
#outfilename = sys.argv[2]        
#print("done features",len(new_features))    
#with open(outfilename,'wb') as fp:
#         pickle.dump(new_features,fp) 
                            

