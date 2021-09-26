import pickle
import random
import sys
import numpy as np
filename = sys.argv[1]
with open(filename,'rb') as f:
     features = pickle.load(f)
new_features = []

count = 0
for feature in features:
      dialogue_final,scene_thisround,did,id_final,type_final,bbox_final,label_final,image_feature = feature
      bbox_final = np.array(bbox_final)
      label_final = np.array(label_final)
      image_feature = np.array(image_feature)
      if len(bbox_final) != len(label_final):
           bboxshape = np.array(bbox_final).shape[0]
           imageshape = np.array(image_feature).shape[0]
           if (bboxshape != imageshape):
              print("find ==========error bbox and label",bboxshape,imageshape)
              continue
      if len(type_final) != len(label_final):
           print("find ==========error typeand label")
           continue
          
      if len(bbox_final)+1 != len(image_feature):
           print("find ==========error future",bbox_final.shape,image_feature.shape)
           print(len(bbox_final),len(image_feature))
           continue
      pos = []
      neg = []
      for i in range(len(label_final)):
             if label_final[i] == 1:
                  pos.append(i)
      #if len(label_final) == 50:
      #   continue
      if len(pos) == 0:
           continue

      for index in pos:
         for p in range(3):
           for k in range(10):
              neg_index = random.randint(0,len(label_final)-1)
              if neg_index not in pos and neg_index not in neg:
                 neg.append(neg_index)
                 break
     
      #print("len of neg and pos",len(neg),len(pos))
      for index in pos:
             id_new =  id_final[index]
             type_new = type_final[index]
             bbox_new = bbox_final[index]
             label_new = label_final[index]
             
             image_new = np.array([image_feature[0]] + [image_feature[index+1]])
             did_new = int(did) * 1000 + index
             #print(image_new.shape)
             #print(image_new)
             new_features.append((dialogue_final,scene_thisround,did_new,id_new,type_new,bbox_new,label_new,image_new))
      #print(did)                            
      #print(scene_thisround)                            
                
      for index in neg:
                 
             id_new =  id_final[index]
             type_new = type_final[index]
             bbox_new = bbox_final[index]
             label_new = label_final[index]
             did_new = int(did) * 1000 + index
             image_new = np.array([image_feature[0]] + [image_feature[index+1]])
             new_features.append((dialogue_final,scene_thisround,did_new,id_new,type_new,bbox_new,label_new,image_new))
      count = count + 1
      #if count > 10:
      #   break          
outfilename = sys.argv[2]        
print("done features",len(new_features))    
with open(outfilename,'wb') as fp:
         pickle.dump(new_features,fp) 
                            
print(len(new_features))
