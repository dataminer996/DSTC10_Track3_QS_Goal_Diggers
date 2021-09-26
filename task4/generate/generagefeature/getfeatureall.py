import utils_f
import pickle
import random
import sys
import numpy as np
import os
filename = sys.argv[1]
allimagefea = []
with open(filename,'rb') as f:
     features = pickle.load(f)
     #print(features[0])
     
     for path,filename in features:
         #print(path)
         imagefilename = "img/obj_pick_one/" + filename + ".png"  
         if os.path.exists(imagefilename):
                pass
         else:
                print(path,imagefilename)
         img_fea = utils_f.readimagefrombin_feature(imagefilename, [[0,0,5,5]])
         allimagefea.append((path,np.array(img_fea[0])))

with open("allgoogdfea.bin", 'wb') as f:
        pickle.dump(allimagefea, f)

 
