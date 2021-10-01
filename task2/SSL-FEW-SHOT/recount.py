import os
import sys
label_dic = {}

with open(sys.argv[1],'r') as tfr:
#   with open(sys.argv[2],'w') as tfw:
      lines = tfr.readlines()
      for line in lines:
        #  print(line)
          a,label =line.split(',')
          if label in label_dic.keys():
              label_dic[label] = label_dic[label] + 1            
          else:
              label_dic[label] = 1
print(label_dic) 

for label in label_dic.keys():
#   print(label)
   if lael.istitle():
        print("label is furt",label)
        i  
#with open(sys.argv[1],'r') as tfr:
#   with open(sys.argv[2],'w') as tfw:
#      lines = tfr.readlines()
#      for line in lines:
#        #  print(line)
#          a,label =line.split(',')
##          if label in label_dic.keys():
#              if label_dic[label] < 40 or label[0].isupper():
                     
 #                 pass
                  #  copynum = int(100/label_dic[label]) + 1
                  #  for i in range(copynum):
                  #      tfw.writelines(line) 
 #             else:
 #                       tfw.writelines(line) 

                    
                           

 
