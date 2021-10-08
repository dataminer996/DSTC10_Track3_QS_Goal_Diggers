import pickle
import sys
import os
import copy
import json
def read_gen_result(filename,modelid):
  with open(filename,'r') as tf:
   lines = tf.readlines()
   results = {}
   for line in lines:
      did,jsontxt = line.split("\t")
      jsondata = json.loads(jsontxt)
      caption = jsondata[0]['caption']
      did = jsondata[0]['img_key']
      did = did*100 + modelid
      results[did] = caption
   return results

def read_binfile(filename):
        with open(filename,'rb') as f:
             features = pickle.load(f)
             #random.shuffle(self.feature)
        print("train example num",len(features))
        return features
        
def find_resulefile(alldir):
  allfiles = os.listdir(alldir)
  modeldirs = []
  for filename in allfiles:
         checkpointdirs = os.listdir(alldir +  "/" + filename)
         for checkpointdir in checkpointdirs:
              ret1 = checkpointdir.find("checkpoint")
              if ret1 >= 0:
                 pytorchfiles =  os.listdir(alldir + "/" + filename + "/" + checkpointdir)
                 for pytorchfile in pytorchfiles:
                       if pytorchfile == 'pytorch_model.bin':
                          if os.path.exists(alldir + "/" + filename + "/" + checkpointdir + "/resulefile.txt"):
                               modeldirs.append(alldir + "/" + filename + "/" + checkpointdir + "/resulefile.txt")

  return modeldirs 
features =  read_binfile(sys.argv[1])
newfeatures = []

gendevtest_dir = sys.argv[2]
resultfilenames  = find_resulefile(gendevtest_dir)
print(resultfilenames)
with open("genresulefilename.bin",'wb') as tf:
      pickle.dump(resultfilenames,tf) 

for i in range(len(resultfilenames)):
     results = read_gen_result(resultfilenames[i],i)
     for feature in features:
               dialogue_final, scene_thisround,dialog_id, id_final, type_final, bbox_final, slotvalue_final, image_feature = feature
               dialogue_final_new = copy.deepcopy(dialogue_final)
               
               try:
                 nes_caption = results[dialog_id*100+i]
               except:
                 print("can't the find the did_index",dialog_id)
                 continue
               #print(nes_caption)
               del dialogue_final_new[-1]                    
               dialogue_final_new.append(nes_caption)
               dialog_id =  dialog_id*100+i
                  # print(dialogue_final[-1])
               newfeatures.append((dialogue_final_new, scene_thisround,dialog_id, id_final, type_final, bbox_final, slotvalue_final, image_feature,0,0))
                   
                    
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
          


