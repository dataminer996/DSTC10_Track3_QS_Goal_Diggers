import pickle
import random
import sys
import numpy as np
filename = sys.argv[1]
with open(filename,'rb') as f:
     features = pickle.load(f)
new_features = []

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
      #   continue

      for i in range(len(label_final)):
             if label_final[i] == 1:
                  pos.append(i)
      if len(pos) == 0:
           continue


      startid = 0
      endid = 30
      num = int(len(label_final) / 30)
      
      if len(label_final) > 30:
         for i in range(num):
             label_new = label_final[startid:endid]
             bbox_new = bbox_final[startid:endid] 
             image_new = np.array([image_feature[0]] + [image_feature[startid+1:endid+1]])
             did_new = did * 1000 +  startid
             type_new = type_final[startid:endid]
             label_new = label_final[startid:endid]
             id_new = id_final[startid:endid]

             
             type_index_new = type_index[startid:endid]


             new_features.append((dialogue_final,scene_thisround,did_new,id_new,type_new,bbox_new,label_new,image_new))
             startid = startid + 30
             endid = endid + 30
         if (len(label_final) % 30) != 0:
             label_new = label_final[startid:]
             bbox_new = bbox_final[startid:]  
             image_new = np.array([image_feature[0]] + [image_feature[startid+1:]])
             did_new = did * 1000 +  startid
             type_new = type_final[startid:]
             label_new = label_final[startid:]
             id_new = id_final[startid:]
             new_features.append((dialogue_final,scene_thisround,did_new,id_new,type_new,bbox_new,label_new,image_new))
      else:
             did_new = did * 1000 +  startid
             new_features.append((dialogue_final,scene_thisround,did_new,id_final,type_final,bbox_final,label_final,image_feature))
    
                     
 
outfilename = sys.argv[2]        
print("done features",len(new_features))    
with open(outfilename,'wb') as fp:
         pickle.dump(new_features,fp) 
                            
print(len(new_features))
