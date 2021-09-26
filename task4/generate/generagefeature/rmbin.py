import sys
import os
for i in range(21):
  cmdline =   'rm ../scene_graph_benchmark_tl_' +  str(i) +  '/*'  + str(sys.argv[1]) + "*.bin  -fr"
  print(cmdline)
  os.system(cmdline) 
