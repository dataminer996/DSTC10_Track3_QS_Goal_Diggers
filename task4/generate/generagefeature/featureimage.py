import pickle
import random
import sys
import numpy as np
filename = sys.argv[1]
with open(filename,'rb') as f:
     features = pickle.load(f)
     print(features[0])
