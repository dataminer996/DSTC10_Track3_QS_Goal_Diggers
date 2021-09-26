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
     print(features[0])
     print(features[0][0])
     #print(features[0][1].shape)
     


 
