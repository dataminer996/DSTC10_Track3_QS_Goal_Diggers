import pickle
import random
import sys
import numpy as np
filename = sys.argv[1]
import copy

with open(filename,'rb') as f:
     filenamelist = pickle.load(f)
new_features = []

allresults = []

for i in range(len(filenamelist)):
  filename = filenamelist[i]
  print(filename)
  with open(filename,'r') as tf:
    lines = tf.readlines()
    newlines = []
    for line in lines:
      did,_ = line.split('\t')
      newlines.append((int(did),line))
      #print(did)
    newresult = copy.deepcopy(newlines)
    print("filename",len(newresult))
    del newlines
  allresults.append(newresult)  
  

feature_did_ids = []



print("len of newfeatures ",len(new_features))

resultfilename = sys.argv[2]
with open(resultfilename,'rb') as f:
    result_features = pickle.load(f)
new_result = []

for  did,pred,label,logitsoutput,idt in result_features:
     prob = logitsoutput.numpy()[1]
     did_new = int(did.item() / 100)
     idt = int(did.item() % 100)
     label = label.item()
     new_result.append((did_new,pred,label,prob,idt))
new_result_sorted = sorted(new_result, key=lambda x: (x[0]))     
dids,preds,labels,probs,ids =  map(list,zip(*new_result_sorted)) 
    
print(new_result_sorted[:20])
lastdid = -1
did_temp_list = []
new_result = []
final_result = []

newresultfp = open(sys.argv[3],'w')

for did,pred,label,prob,idt in new_result_sorted:
     if did == lastdid or lastdid == -1:
         did_temp_list.append((did,prob,label,idt))
         lastdid = did  
     else:
         a,b,c,d = map(list,zip(*did_temp_list))
         
         did_temp_list_sort =  sorted(did_temp_list, key=lambda x: (x[1]),reverse=True)
         
         id_new = did_temp_list_sort[0][3]
         did_new =   did_temp_list_sort[0][0]
         find = 0
         for did_old,line_old in   allresults[id_new]:  
            if did_old == did_new:
               print("find right",did_old,did_new)
               newresultfp.writelines(line_old)
               find = 1
         if find == 0:
           print("find error",did_new,id_new)
                          
         del(did_temp_list_sort)
         del(did_temp_list)
         
         did_temp_list = []
         did_temp_list.append((did,prob,label,idt))
         lastdid = did
did_temp_list_sort =  sorted(did_temp_list, key=lambda x: (x[1]),reverse=True)
a,b,c,d = map(list,zip(*did_temp_list))
id_new = did_temp_list_sort[0][3]
did_new =   did_temp_list_sort[0][0]

for did_old,line_old in   allresults[id_new]:
   if did_old == did_new:
      newresultfp.writelines(line_old)



               
