import sys
import os

imagedir = sys.argv[1]
imagelist = os.listdir(imagedir)
csvfile = sys.argv[2]
with open(csvfile,'w') as tfw:
    for imagename in imagelist: 
        tfw.writelines(imagename + ",None\n") 
