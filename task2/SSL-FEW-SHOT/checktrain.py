import os
import sys
with open(sys.argv[1],'r') as tfr:
      lines = tfr.readlines()
      print(len(lines))
      for line in lines:
        #  print(line)
          a =line.split(',')
          if len(a) == 2:
  #         print(a[0],a[1])
                  filename = sys.argv[2] + "/" + a[0] 
                  if not os.path.exists(filename):
                      print("don't find line",line)
          else:
               print("label line error",line)
