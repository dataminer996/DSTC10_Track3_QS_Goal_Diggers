import pickle
import random
import sys
import numpy as np
from sklearn import metrics
import json
import os 
#filename = sys.argv[1]


def thre_modify(thre,results):
     new_results = []
     new_preds = []
     new_labels = []
     #print("result of thre",thre)
     for did,pred,label,logitsoutput,_  in results:
        if logitsoutput.numpy()[1] > thre:
          new_preds.append(1)
        else:
          new_preds.append(0)
        new_labels.append(label)
     ret = metrics.classification_report(list(new_labels),new_preds,target_names=None,digits=4,output_dict=True)
     #print(ret)
     return ret


def thre_old_modify(thre,results):
     new_results = []
     new_preds = []
     new_labels = []
     #print("result of thre",thre)
     for did,pred,label,logitsoutput  in results:
        if logitsoutput.numpy()[1] > thre:
          new_preds.append(1)
        else:
          new_preds.append(0)
        new_labels.append(label)
     ret = metrics.classification_report(list(new_labels),new_preds,target_names=None,digits=4,output_dict=True)
     #print(ret)
     return ret     
def update_result(filename):
      with open(filename,'rb') as f:
           results = pickle.load(f)
     # print(results[0])
      
      did_list = []
      
      
      #print("results",len(results))
      ret = thre_modify(0.5,results)
      print(filename+"old",ret['1']['precision'],ret['1']['recall'],ret['1']['f1-score'])
      
      #for thre in range(1,10):
      #   thre_modify(float(thre)/1000 + 0.99,results)
      for did_index,pred,label,logitsoutput,_  in results:
          did = did_index / 1000
          did_list.append((did,logitsoutput.numpy()[1],label,did_index))
      #print("did_list",len(did_list))
      did_list_sorted = sorted(did_list, key=lambda x: (x[0]))
      
      lastdid = -1
      did_temp_list = []
      new_result = []
      final_result = []
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
                          new_result.append((did_temp_list_sort[i][0],did_temp_list_sort[i][1],did_temp_list_sort[i][2],did_temp_list_sort[i][3],1,1,obj_num))
                      else:
                          new_result.append((did_temp_list_sort[i][0],did_temp_list_sort[i][1],did_temp_list_sort[i][2],did_temp_list_sort[i][3],0,0,obj_num))
                      
               del(did_temp_list_sort)
               del(did_temp_list)
               
               did_temp_list = []
               did_temp_list.append((did,prob,label,did_index))
               lastdid = did
      
      
      did_temp_list_sort =  sorted(did_temp_list, key=lambda x: (x[1]),reverse=True)
      a,b,c,d = map(list,zip(*did_temp_list))
      obj_num = 0
      for label_temp in c:
            if label_temp==1:
                obj_num = obj_num + 1
      
      for i in range(len(did_temp_list_sort)):
             if i < obj_num:
                 new_result.append((did_temp_list_sort[i][0],did_temp_list_sort[i][1],did_temp_list_sort[i][2],did_temp_list_sort[i][3],1,1,obj_num))
             else:
                 new_result.append((did_temp_list_sort[i][0],did_temp_list_sort[i][1],did_temp_list_sort[i][2],did_temp_list_sort[i][3],0,0,obj_num))
        
      
      #print("new_result",len(new_result))    
      sus = 0
      for i in range(len(new_result)):
         if new_result[i][2] == new_result[i][4]:
             sus = sus + 1
      #print("acc",sus,len(new_result),float(100*sus)/len(new_result))
      did,prob,last_label,did_index,a,last_pred,_ = map(list,zip(*new_result))
      ret = metrics.classification_report(last_label,last_pred,target_names=None,digits=4,output_dict=True)
      
      return ret['1']['precision'],ret['1']['recall'],ret['1']['f1-score'],new_result


def update_result_objid(filename,obj_num_dic):
      with open(filename,'rb') as f:
           results = pickle.load(f)
     # print(results[0])

#      obj_num_dids,obj_num_lists = map(list,zip(*obj_num_list))
     
 
      did_list = []

     
      #print("results",len(results))
      ret = thre_modify(0.5,results)
      print(filename+"old",ret['1']['precision'],ret['1']['recall'],ret['1']['f1-score'])
    
      #for thre in range(1,10):
      #   thre_modify(float(thre)/1000 + 0.99,results)
      for did_index,pred,label,logitsoutput,objid  in results:
          did = did_index / 1000
          #did_index = int(did.item())
          did_list.append((did.item(),logitsoutput.numpy()[1],label,did_index,objid))
      #print("did_list",len(did_list))
      did_list_sorted = sorted(did_list, key=lambda x: (x[0]))

      didold,_,_,_,_ = map(list,zip(*did_list_sorted))
      
      did_num = len(list(set(didold)))
#      print("inputresult did_num",did_num)
      
      lastdid = -1
      did_temp_list = []
      new_result = [] 
      obj_zero_num = 0
      obj_num_total = 0
      for did,prob,label,did_index,objid in did_list_sorted :
           if did == lastdid or lastdid == -1:
               did_temp_list.append((did,prob,label,did_index,objid))
               lastdid = did
               try:
                obj_num = obj_num_dic[str(did)]
               except:
                   print("error",did)
                   obj_num = 0  
           else:
               a,b,c,d,e = map(list,zip(*did_temp_list))
               ##obj_num = 0
               #for label_temp in c:
               ##    if label_temp==1:
               #       obj_num = obj_num + 1
               try:
                  obj_num = obj_num_dic[str(did)]
               except:
                   print("error",did)
                   obj_num1 = 0  
               obj_num_total = obj_num_total + obj_num
               did_temp_list_sort =  sorted(did_temp_list, key=lambda x: (x[1]),reverse=True)
               if obj_num == 0:
                      obj_zero_num = obj_zero_num + 1            

               for i in range(len(did_temp_list_sort)):
                      if i < obj_num:
                          new_result.append((did_temp_list_sort[i][0],did_temp_list_sort[i][1],did_temp_list_sort[i][2],did_temp_list_sort[i][3],did_temp_list_sort[i][4],1,obj_num))
                      else:
                          new_result.append((did_temp_list_sort[i][0],did_temp_list_sort[i][1],did_temp_list_sort[i][2],did_temp_list_sort[i][3],did_temp_list_sort[i][4],0,obj_num))

               del(did_temp_list_sort)
               del(did_temp_list)

               did_temp_list = []  
               did_temp_list.append((did,prob,label,did_index,objid))
               lastdid = did


      did_temp_list_sort =  sorted(did_temp_list, key=lambda x: (x[1]),reverse=True)
      a,b,c,d,e = map(list,zip(*did_temp_list))
      #obj_num = 0
      #for label_temp in c:
      #      if label_temp==1:
      #          obj_num = obj_num + 1

      #obj_num = int(obj_num_list[obj_num_dids.index(did_index)])
      #try:
      #  obj_num = obj_num_dic[str(did_index)]
      #except:
      #        print("error",did)
      #        obj_num = 0  
      if obj_num == 0:
                      obj_zero_num = obj_zero_num + 1            
      obj_num_total = obj_num_total + obj_num
      for i in range(len(did_temp_list_sort)):
             if i < obj_num:
                 new_result.append((did_temp_list_sort[i][0],did_temp_list_sort[i][1],did_temp_list_sort[i][2],did_temp_list_sort[i][3],did_temp_list_sort[i][4],1,obj_num))
             else:
                 new_result.append((did_temp_list_sort[i][0],did_temp_list_sort[i][1],did_temp_list_sort[i][2],did_temp_list_sort[i][3],did_temp_list_sort[i][4],0,obj_num))

     
      #print("new_result",len(new_result))    
      sus = 0
      for i in range(len(new_result)):
         if new_result[i][2] == new_result[i][5]:
             sus = sus + 1
      #print("acc",sus,len(new_result),float(100*sus)/len(new_result))
      did,prob,last_label,did_index,objid,last_pred,_ = map(list,zip(*new_result))
      ret = metrics.classification_report(last_label,last_pred,target_names=None,digits=4,output_dict=True)
      did_num = len(list(set(did)))
      print(ret)
      #did_num = set(did)
#      print("newresult obj did_num",did_num,obj_zero_num,obj_num_total)
      return ret['1']['precision'],ret['1']['recall'],ret['1']['f1-score'],new_result



def update_old_result(filename):
      with open(filename,'rb') as f:
           results = pickle.load(f)
      
      did_list = []
      
      for did_index,pred,label,logitsoutput  in results:
          did = did_index / 1000
          did_list.append((did,logitsoutput.numpy()[1],label,did_index))
      #print("did_list",len(did_list))
      did_list_sorted = sorted(did_list, key=lambda x: (x[0]))
      
      lastdid = -1
      did_temp_list = []
      new_result = []
      obj_zero_num = 0
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
                          new_result.append((did_temp_list_sort[i][0],did_temp_list_sort[i][1],did_temp_list_sort[i][2],did_temp_list_sort[i][3],1,1,obj_num))
                      else:
                          new_result.append((did_temp_list_sort[i][0],did_temp_list_sort[i][1],did_temp_list_sort[i][2],did_temp_list_sort[i][3],0,0,obj_num))
               if obj_num == 0:
                      obj_zero_num = obj_zero_num + 1            
               del(did_temp_list_sort)
               del(did_temp_list)
               
               did_temp_list = []
               did_temp_list.append((did,prob,label,did_index))
               lastdid = did
      
      
      did_temp_list_sort =  sorted(did_temp_list, key=lambda x: (x[1]),reverse=True)
      a,b,c,d = map(list,zip(*did_temp_list))
      obj_num = 0
      for label_temp in c:
            if label_temp==1:
                obj_num = obj_num + 1
      if obj_num == 0:
                      obj_zero_num = obj_zero_num + 1            
      #do it for last did
      for i in range(len(did_temp_list_sort)):
             if i < obj_num:
                 new_result.append((did_temp_list_sort[i][0],did_temp_list_sort[i][1],did_temp_list_sort[i][2],did_temp_list_sort[i][3],1,1,obj_num))
             else:
                 new_result.append((did_temp_list_sort[i][0],did_temp_list_sort[i][1],did_temp_list_sort[i][2],did_temp_list_sort[i][3],0,0,obj_num))
        
      
      #print("new_result",len(new_result))    
      sus = 0
      for i in range(len(new_result)):
         if new_result[i][2] == new_result[i][4]:
             sus = sus + 1
      #print("acc",sus,len(new_result),float(100*sus)/len(new_result))
      did,prob,last_label,did_index,a,last_pred,_ = map(list,zip(*new_result))
     # ret = metrics.classification_report(last_label,last_pred,target_names=None,digits=4,output_dict=True)
    #  print(ret)      
      return new_result



obj_num_file = sys.argv[2]

with open(obj_num_file) as f:
        num_data = json.load(f)

lldir = sys.argv[1]
allfiles = os.listdir(alldir)
modeldirs = []
for filename in allfiles:
         checkpointdirs = os.listdir(alldir +  "/" + filename)
         for checkpointdir in checkpointdirs:
              ret1 = checkpointdir.find("checkpoint")
              if ret1 >= 0:
                 pytorchfiles =  os.listdir(alldir + "/" + filename + "/" + checkpointdir)
                 for pytorchfile in pytorchfiles:
                       if pytorchfile == 'pytorch_model.bin':
                            if os.path.exists(alldir + "/" + filename + "/" + checkpointdir + "/resultfromsys.bin"):
                               modeldirs.append(alldir +  "/" + filename + "/" + checkpointdir + "/resultfromsys.bin")

probs = []
preds = []
for modeldir in modeldirs: 
      new_result = update_result_objid(modeldir,num_data)
      did,prob,last_label,did_index,a,last_pred,_ = map(list,zip(*new_result))
      probs.append(prob)
      preds.append(last_pred) 
           
for i in range(len(preds[0])):
     pred_num = None
     for j in range(len(preds)):
        if pred_num is None:
          pred_num = one_hot(preds[j][i],len(pred_type_list))
        else:
          pred_num = pred_num + one_hot(preds[j][i],len(pred_type_list))
     if checkmax(pred_num) == 1:
          results.append(np.argmax(pred_num))
          continue
     prob_total = None
     for j in range(len(probs)):
        if prob_total is None:
           prob_total = np.array(probs[j][i])
        else:
           prob_total = np.array(probs[j][i]) + prob_total

     results.append(np.argmax(prob_total))


  
flag = 1
with open(modeldir + "update",'wb') as fp:
      pickle.dump(ret[3],fp)                
  
                          


 
    
    
    
    

       


  
