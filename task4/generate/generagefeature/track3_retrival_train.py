import pickle
import sys
import os
import copy

def read_binfile(filename):
        with open(filename,'rb') as f:
             features = pickle.load(f)
             #random.shuffle(self.feature)
        print("train example num",len(features))
        return features
        
features =  read_binfile(sys.argv[1])
newfeatures = []
newfeatures_1 = []

nes_dir = sys.argv[2]
nes_filenames = os.listdir(nes_dir)
nes_examples = []
for feature in features:
        dialogue_final, scene_thisround,dialog_id, id_final, type_final, bbox_final, slotvalue_final, image_feature = feature
        #print(dialogue_final)
        newfeatures.append((dialogue_final, scene_thisround,dialog_id, id_final, type_final, bbox_final, slotvalue_final, image_feature,1))

for nes_filename in nes_filenames:
    nespath = os.path.join(nes_dir,nes_filename)
    print(nespath)
    print("done features",len(newfeatures))    
    with open(nespath,'r') as tf:
       nes_example = []
       dids = []
       lines = tf.readlines()
       for line in lines:
           line = line.strip()
           did,caption = line.split('\t',1)
           #print(caption)
           nes_example.append(caption.strip())           
           dids.append(int(did))
       for feature in features:
               dialogue_final, scene_thisround,dialog_id, id_final, type_final, bbox_final, slotvalue_final, image_feature = feature
               dialogue_final_new = copy.deepcopy(dialogue_final)
               
               try:
                 did_index = dids.index(int(dialog_id))
               except:
                 print("can't the find the did_index",dialog_id)
                 continue
               nes_caption = nes_example[did_index]
               #print(nes_caption)
               pos_caption = dialogue_final_new[-1:]
               if nes_caption != pos_caption:  
                   del dialogue_final_new[-1]                    
                   dialogue_final_new.append(nes_caption)
              
                  # print(dialogue_final[-1])
                   newfeatures.append((dialogue_final_new, scene_thisround,dialog_id, id_final, type_final, bbox_final, slotvalue_final, image_feature,0))
                   
                    
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
          


