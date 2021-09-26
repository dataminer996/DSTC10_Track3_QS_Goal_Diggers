import pickle
import random
import sys
import numpy as np
from sklearn import metrics

filename = sys.argv[1]
with open(filename,'rb') as f:
     new_result = pickle.load(f)
#new_features = []

did,prob,last_label,did_index,a,last_pred = map(list,zip(*new_result))
ret = metrics.classification_report(last_label,last_pred,target_names=None,digits=4,output_dict=False)
print(ret)

exit()

for feature in features:
      dialogue_final,scene_thisround,id_final,type_final,bbox_final,label_final,image_feature = feature
      pos = []
      neg = []
      for i in range(len(label_final)):
             if label_final[i] == 1:
                  pos.append(i)
      if len(label_final) == 50:
         continue
      if len(pos) == 0:
           continue

      for index in pos:
         for p in range(3):
           for k in range(10):
              neg_index = random.randint(0,len(label_final)-1)
              if neg_index not in pos and neg_index not in neg:
                 neg.append(neg_index)
                 break
     
      print("len of neg and pos",len(neg),len(pos))
      for index in pos:
             id_new =  id_final[index]
             type_new = type_final[index]
             bbox_new = bbox_final[index]
             label_new = label_final[index]
             
             image_new = np.array([image_feature[0]] + [image_feature[index+1]])
             #print(image_new.shape)
             #print(image_new)
             new_features.append((dialogue_final,scene_thisround,id_new,type_new,bbox_new,label_new,image_new))
                            
                
      for index in neg:
             id_new =  id_final[index]
             type_new = type_final[index]
             bbox_new = bbox_final[index]
             label_new = label_final[index]
             image_new = np.array([image_feature[0]] + [image_feature[index+1]])
             new_features.append((dialogue_final,scene_thisround,id_new,type_new,bbox_new,label_new,image_new))
                
#outfilename = sys.argv[2]        
#print("done features",len(new_features))    
#with open(outfilename,'wb') as fp:
#         pickle.dump(new_features,fp) 
                            

