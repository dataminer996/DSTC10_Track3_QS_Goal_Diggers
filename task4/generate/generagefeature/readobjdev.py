import pickle
import random
import sys
import numpy as np
filename = sys.argv[1]
with open(filename,'rb') as f:
     features = pickle.load(f)
new_features = []

dialogue_final,scene_thisround,dids,id_final,type_final,bbox_final,label_final,image_feature =  map(list,zip(*features))


didnew = []
for did in dids:
   #newid = int(did/1000)
   newid = int(did)
   if newid not in didnew:
      # print(newid)
       didnew.append(newid)

print("did all num",len(didnew))
print("did set num",len(list(set(didnew))))

for feature in features:
    
      dialogue_final,scene_thisround,did,id_final,type_final,bbox_final,label_final,image_feature = feature
      if int(did) == int(sys.argv[2]):
#      pint(image_feature.shape,label_final,label_final.shape)
           print(label_final)
           print(id_final)
      continue
      #print(did)
      #if len(bbox_final) != len(label_final):
      #     print("find ==========error bbox and label")
      #if len(type_final) != len(label_final):
      #     print("find ==========error typeand label")
          
      #if len(bbox_final)+1 != len(image_feature):
      #     print("find ==========error future")
      #     print(len(bbox_final),len(image_feature))
      break
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
      if len(bbox_final)+1 != len(image_feature):
           print("find ==========error future")
           print(len(bbox_final),len(image_feature))
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
             if len(label_final)+1 != len(image_feature):
                 print("find ==========error future")
                 print(len(label_final),len(image_feature))
                 print(index,len(image_feature))
             new_features.append((dialogue_final,scene_thisround,id_new,type_new,bbox_new,label_new,image_new))
                
#outfilename = sys.argv[2]        
#print("done features",len(new_features))    
#with open(outfilename,'wb') as fp:
#         pickle.dump(new_features,fp) 
                            

