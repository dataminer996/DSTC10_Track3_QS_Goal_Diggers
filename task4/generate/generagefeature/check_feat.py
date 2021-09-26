import json
import base64
import numpy as np


with open('./test.feat') as f:
    data = f.readlines()[3:]
row = data[0].split('\t')
print(row[0])

num_boxes = int(row[1])
features = np.frombuffer(base64.b64decode(row[-1]),dtype=np.float32).reshape(num_boxes,-1)
#for  feature in features:
	#       print(feature[:6])
#       print(feature[-6:])
print(features[0])
np.savetxt('dog_ori.txt',features[0])
file_path = 'output/X152C5_test/inference/vinvl_vg_x152c4/predictions.tsv'
with open(file_path,'r') as f:
    data = f.readlines()
    # j = json.load(f)
for row in data:
    print(row.split('\t')[0])
    if "COCO_test2014_000000000027" == row.split('\t')[0]:
        pass
    else:
        continue
    con = row.split('\t')[-1]
    # print(con)
    con = json.loads(con)
    ob_list = con.get('objects')
    count = 0
    for ob in ob_list:
        # print(ob)
        feat = ob.get('feature')
        my_feature = np.frombuffer(base64.b64decode(feat),dtype=np.float32).reshape((-1))
        if ob.get('class') == 'dog':
            print('my: ',my_feature)
            np.savetxt('minux.txt',features[0][:-6] - my_feature)
            np.savetxt('dog_gen.txt',my_feature)
        find = 0
        for feature in features:
            #print('split: ',feature)
            for i in range(6):
	    #                print(feature[:6])
	    #    print(feature[-6:])
		#print(feature.shape)
		#print('split_i: ',feature[i:])
		#print(feature[i:i+2048].shape)
		#print(my_feature.shape)
                if list(my_feature) == list(feature[i:i+2048]):
                    find += 1
                    break
        if find > 0:
            print('find num: ',find)
            
        count += 1

    print(count)
