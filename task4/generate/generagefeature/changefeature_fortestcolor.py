import pickle
import random
import sys
import numpy as np
filename = sys.argv[1]
with open(filename,'rb') as f:
     features = pickle.load(f)
new_features = []

def one_hot(_train_labels,no_of_different_labels):
   _train_labels = np.array(_train_labels)
   train_labels = np.zeros((_train_labels.shape[0],no_of_different_labels))
   train_labels[np.arange(_train_labels.shape[0]),_train_labels-1] = 1
   return train_labels

def add_one_hot(one_hots):
  start = 0
  one_hot_add = None
  for one_hot in one_hots:
     if start == 0:
        one_hot_add = one_hot
        start = start + 1
     else:
        one_hot_add = one_hot_add + one_hot
        start = start + 1

  return one_hot_add


for feature in features:
#      dialogue_final,scene_thisround,did,id_final,type_final,bbox_final,label_final,image_feature = feature
      dialogue_final,scene_thisround,did,id_final,type_final,bbox_final,label_final,image_feature,prefab_new,type_index,color,object_num = feature

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
             else:
                  neg.append(i)
      #if len(label_final) == 50:
      #   continue
      if len(pos) == 0:
           continue
      
 
      #print("len of neg and pos",len(neg),len(pos))
      for index in pos:
             id_new =  id_final[index]
             type_new = type_final[index]
             bbox_new = bbox_final[index]
             label_new = label_final[index]
             did_new = int(did) * 1000 + index 
             color_new = color[index]
             color_index = add_one_hot(one_hot(color_new,48))
             type_index_new = type_index[index]

             image_new = np.array([image_feature[0]] + [image_feature[index+1]])
             new_features.append((dialogue_final,scene_thisround,did_new,id_new,type_new,bbox_new,label_new,image_new,color_index,type_index_new,object_num))


#             image_new = np.array([image_feature[0]] + [image_feature[index+1]])
             #print(image_new.shape)
             #print(image_new)
 #            new_features.append((dialogue_final,scene_thisround,did_new,id_new,type_new,bbox_new,label_new,image_new))
                            
                
      for index in neg:
                 
             id_new =  id_final[index]
             type_new = type_final[index]
             bbox_new = bbox_final[index]
             did_new = int(did) * 1000 + index 
             label_new = label_final[index]
             #image_new = np.array([image_feature[0]] + [image_feature[index+1]])
             #new_features.append((dialogue_final,scene_thisround,did_new,id_new,type_new,bbox_new,label_new,image_new))
             color_new = color[index]
             color_index = add_one_hot(one_hot(color_new,48))
             type_index_new = type_index[index]

             image_new = np.array([image_feature[0]] + [image_feature[index+1]])
             new_features.append((dialogue_final,scene_thisround,did_new,id_new,type_new,bbox_new,label_new,image_new,color_index,type_index_new,object_num))
                
outfilename = sys.argv[2]        
print("done features",len(new_features))    
with open(outfilename,'wb') as fp:
         pickle.dump(new_features,fp) 
                            
print(len(new_features))
