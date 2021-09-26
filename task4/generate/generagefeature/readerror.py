import pickle
import random
import sys
import numpy as np
#filename = sys.argv[1]
for i in range(int(sys.argv[1]),int(sys.argv[2])):
  filename = '../scene_graph_benchmark_tl_' +  str(i) +  "/picture_mistake_all.bin"
  print(filename)
  with open(filename,'rb') as f:
     features = pickle.load(f)
     print(features)                       

