import pickle
import random
import sys
import numpy as np
from sklearn import metrics


filename = sys.argv[1]

with open(filename,'rb') as f:
     results = pickle.load(f)
print(results[0])

did_list = []
def thre_modify(thre,results):
     new_results = []
     new_preds = []
     new_labels = []
     print("result of thre",thre)
     for did,pred,label,logitsoutput,_  in results:
        if logitsoutput.numpy()[1] > thre:
          new_preds.append(1)
        else:
          new_preds.append(0)
        new_labels.append(label)
     ret = metrics.classification_report(list(new_labels),new_preds,target_names=None,digits=4)
     print(ret)
     return ret

print("results",len(results))
thre_modify(0.5,results)

#for thre in range(1,10):
#   thre_modify(float(thre)/1000 + 0.99,results)
for did_index,pred,label,logitsoutput,_  in results:
    did = did_index / 1000
    did_list.append((did,logitsoutput.numpy()[1],label,did_index))
print("did_list",len(did_list))
did_list_sorted = sorted(did_list, key=lambda x: (x[0]))

lastdid = -1
did_temp_list = []
new_result = []
for did,prob,label,did_index in did_list_sorted :
     if did == lastdid or lastdid == -1:
         did_temp_list.append((did,prob,label,did_index))
         lastdid = did  
     else:
         a,b,c,d = map(list,zip(*did_temp_list))
         obj_num = 0
         for label_temp in c:
             if label_temp==1:
                obj_num = obj_num + 1
         did_temp_list_sort =  sorted(did_temp_list, key=lambda x: (x[1]),reverse=True)
         
         for i in range(len(did_temp_list_sort)):
                if i < obj_num:
                    new_result.append((did_temp_list_sort[i][0],did_temp_list_sort[i][1],did_temp_list_sort[i][2],did_temp_list_sort[i][3],1))
                else:
                    new_result.append((did_temp_list_sort[i][0],did_temp_list_sort[i][1],did_temp_list_sort[i][2],did_temp_list_sort[i][3],0))
                
         del(did_temp_list_sort)
         del(did_temp_list)
         
         did_temp_list = []
         did_temp_list.append((did,prob,label,did_index))
         lastdid = did


did_temp_list_sort =  sorted(did_temp_list, key=lambda x: (x[1]),reverse=True)

for i in range(len(did_temp_list_sort)):
       if i < obj_num:
           new_result.append((did_temp_list_sort[i][0],did_temp_list_sort[i][1],did_temp_list_sort[i][2],did_temp_list_sort[i][3],1))
       else:
           new_result.append((did_temp_list_sort[i][0],did_temp_list_sort[i][1],did_temp_list_sort[i][2],did_temp_list_sort[i][3],0))
  

print("new_result",len(new_result))    
sus = 0
for i in range(len(new_result)):
   if new_result[i][2] == new_result[i][4]:
       sus = sus + 1
print("acc",sus,len(new_result),float(100*sus)/len(new_result))
did,prob,last_label,did_index,last_pred = map(list,zip(*new_result))
ret = metrics.classification_report(last_label,last_pred,target_names=None,digits=4,output_dict=True)
dd
print(ret)



 
    
    
    
    

       


  
