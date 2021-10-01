import os
import sys
basename = sys.argv[1]
newdir = sys.argv[2]
dirlist = os.listdir(basename)
for dirname in dirlist:
   filelist = os.listdir(basename + "/" + dirname)
   for filename in filelist:
     cmdline = "/bin/cp "  + "\'" +basename + "/" + dirname + "/" + filename +"\'" + "/*   "   +  newdir + "/."
     print(cmdline)
     os.system(cmdline)

 

