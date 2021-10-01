import os
import json
import sys


dirname = sys.argv[1]
class_dic = {}
with open(dirname,'r') as tf:
      lines = tf.readlines()
      i = 0
      for line in lines:
          if i == 0:
             i = 1
             continue
          _,classname = line.split(',')
          if classname in class_dic.keys():
              class_dic[classname] = class_dic[classname] + 1
          else:
              class_dic[classname] = 1
print(class_dic)
with open(sys.argv[2],'w') as tfw:
 for key in class_dic.keys():
       if class_dic[key] < 50:
            print(key,class_dic[key])            
            continue 
       for line in lines:
          _,classname = line.split(',')
          if classname == key:
              tfw.writelines(line)   

#outputfile = sys.argv[2]
#filename_list,label_list,id_list = create_label(dirname)
#with  open(outputfile,'w') as tf:
#   for i in range(len(filename_list)):
#       label = label_list[i].replace(',','_')
#       label = label.replace(' ','_')
#       tf.writelines(filename_list[i]+","+str(label)+"\n")

#get_type_id_img(sys.argv[1])
