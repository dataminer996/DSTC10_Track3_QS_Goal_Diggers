import pickle
import sys
with open(sys.argv[1],'rb') as f:
             features =  pickle.load(f)
             for feature in features:
                dids,preds,labels,logitsoutput,retri_ids = feature
                print(feature)
                print(dids)
                print(logitsoutput)
                print(retri_ids)
                break

