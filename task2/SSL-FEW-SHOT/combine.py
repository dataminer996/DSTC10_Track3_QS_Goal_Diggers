import sys
import os

with open(sys.argv[1],'r') as tfr:
  with open(sys.argv[2],'r') as tfr2:
   with open(sys.argv[3],'w') as tfw:
      lines = tfr.readlines()
      lines1 = tfr2.readlines()
      for line in lines:
         tfw.writelines(line)
      for line in lines1:
         tfw.writelines(line)

