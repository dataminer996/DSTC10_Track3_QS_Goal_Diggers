import pickle
import sys
import os
import random
import copy

def read_binfile(filename):
        with open(filename,'rb') as f:
             features = pickle.load(f)
             #random.shuffle(self.feature)
        print("dev example num",len(features))
        return features
        
features =  read_binfile(sys.argv[1])
newfeatures = []
newfeatures_1 = []

nes_features =  read_binfile(sys.argv[2])


#nes_examples = []
#for feature in features:
#        dialogue_final, scene_thisround,dialog_id, id_final, type_final, bbox_final, slotvalue_final, image_feature = feature
        #print(dialogue_final)
#        newfeatures.append((dialogue_final, scene_thisround,dialog_id, id_final, type_final, bbox_final, slotvalue_final, image_feature,1))

dialogue_finals, scene_thisrounds,dialog_ids, id_finals, type_finals, bbox_finals, slotvalue_finals, image_features = map(list,zip(*features))

all_num  = 0
for nes_feature in nes_features:       
       index = random.randint(0,9)
       #if index != 1:
       #    continue
       all_num = all_num + 1
       #if all_num > 10:
       #     break  
       num, turn_idx, text, label,retri_index = nes_feature
       did_index = num*100 + turn_idx
       index = dialog_ids.index(did_index)
       
       
       dialogue_final_new = copy.deepcopy(dialogue_finals[index])
       nes_caption = text
       #pos_caption = dialogue_final_new[-1:]
       del dialogue_final_new[-1] 
       dialogue_final_new.append(nes_caption)         
       newfeatures.append((dialogue_final_new, scene_thisrounds[index],dialog_ids[index], id_finals[index], type_finals[index], bbox_finals[index], slotvalue_finals[index], image_features[index],label,retri_index))
                
                    
outfilename = sys.argv[3]        
print("done features",len(newfeatures))    
with open(outfilename,'wb') as fp:
         pickle.dump(newfeatures,fp) 

           #nes_example.a    
        
        
#
#for feature in features:
#     dialogue_final, scene_thisround,dialog_id, id_final, type_final, bbox_final, slotvalue_final, image_feature,label_final = feature[i]
#     
#     dialogue_final[-1:] 
          


