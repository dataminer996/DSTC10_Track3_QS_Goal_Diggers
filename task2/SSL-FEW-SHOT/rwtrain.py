import os
import sys

image_rootdir = sys.argv[3]
with open(sys.argv[1],'r') as tfr:
   with open(sys.argv[2],'w') as tfw:
      lines = tfr.readlines()
      for line in lines:
        #  print(line)
          a =line.split('.png')
          if len(a) == 2:
            #print(a[0])
            if os.path.exists(image_rootdir+ "/" + a[0] + ".jpg"):
                   newname =  a[0].replace(' ','_')
                   newname = newname.replace(",","_")
                   if newname != a[0]:
                        cmdline = "mv  " + "\'" + image_rootdir + "/" + a[0] + ".jpg\' "   + image_rootdir + "/"  + newname + ".jpg "   
                        print(cmdline)
                        os.system(cmdline)  
                        line = line.replace(a[0],newname)
                   line = line.replace('png','jpg')
                   tfw.writelines(line)
            else:
                  
                  print("don't find line",line)
