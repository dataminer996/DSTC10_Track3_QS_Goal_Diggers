import sys
import numpy as np
import os
import pickle
import json
with open(sys.argv[1], 'rb') as f:
        features = pickle.load(f)

print(len(features))
colors = {'dark brown', 'orange', 'light blue', 'dirty green', 'dark blue', 'wooden', 'purple', 'golden', 'dark grey', 'dark red', 'grey', 'light grey', ' yellow', 'dark green', 'white', ' purple', 'black and white', ' white', 'beige', 'green', 'yellow', 'olive', ' olive', 'light red', 'light orange', 'black', 'dark pink', ' brown', ' blue', ' orange', 'red', 'violet', ' black', ' violet', ' pink', 'blue', 'brown', 'dirty grey', 'maroon', 'dark yellow', 'light pink', ' dark blue', 'dark violet', 'pink', ' red', ' green', ' grey', ' light green'}
types = {'CouchChair', 'vest', 'shirt', 'Chair', 'Sofa', 'Shelves', 'suit', 'dress', 'EndTable', 'shirt, vest', 'AreaRug', 'tshirt', 'Bed', 'Lamp', 'jeans', 'blouse', 'trousers', 'Table', 'CoffeeTable', 'tank top', 'hat', 'coat', 'skirt', 'shoes', 'sweater', 'hoodie', 'jacket', 'joggers'}

colors = list(colors)
types = list(types)

fashion_alls = json.load(open('data/fashion_prefab_metadata_all.json','r'))
funiture_alls = json.load(open('data/furniture_prefab_metadata_all.json','r'))

features_new = []
#print(fashion_alls.keys())
#print(funiture_alls.keys())
for feature in features:
   dialogue_final,scene_thisround,diag_id,id_final,type_final,bbox_final,label_final,image_feature,prefab_final = feature
#   print(prefab_final)
   type_index_new = []
   type_new = []
   bbox_new = []
   label_new = []
   image_feature_new = []
   prefab_new = []
   color_new = []
   id_new = []
   image_feature_new.append(image_feature[0])
   for index in range(len(prefab_final)):
      prefab = prefab_final[index]
      if prefab in list(fashion_alls.keys()):
       #  print("fashion",prefab)
         if fashion_alls[prefab]['type'] != type_final[index]:
             print("type error",prefab,diag_id,fashion_alls[prefab]['type'],type_final[index])
             continue
         type_index_new.append(types.index(type_final[index]))
         type_new.append(type_final[index])
         color = fashion_alls[prefab]['color']
         color_list =  color.split(',')
         color_index = []
         for colorsingle in color_list:
              color_index.append(colors.index(colorsingle))

         color_new.append(color_index)
         id_new.append(id_final[index])
         bbox_new.append(bbox_final[index])
         label_new.append(label_final[index])
         image_feature_new.append(image_feature[index+1])
         prefab_new.append(prefab)

      elif prefab in list(funiture_alls.keys()):  

         if funiture_alls[prefab]['type'] != type_final[index]:
             print("type error",prefab,diag_id,funiture_alls[prefab]['type'],type_final[index])
             continue
         type_index_new.append(types.index(type_final[index]))
         type_new.append(type_final[index])
         color = funiture_alls[prefab]['color']
         color_list =  color.split(',')
         color_index = []
         for colorsingle in color_list:
              color_index.append(colors.index(colorsingle))

         color_new.append(color_index)

         id_new.append(id_final[index])
         bbox_new.append(bbox_final[index])
         label_new.append(label_final[index])
         
         image_feature_new.append(image_feature[index+1])
         prefab_new.append(prefab)
      else:
         print("error",prefab,diag_id)
   object_num = 0
   for label in label_new:
       if label == 1:
           object_num = object_num + 1

   features_new.append(( dialogue_final,scene_thisround,diag_id,id_new,type_new,bbox_new,label_new,image_feature_new,prefab_new,type_index_new,color_new,object_num))

print(len(features_new))

with open(sys.argv[2], 'wb') as f:
        pickle.dump(features_new, f)
 
