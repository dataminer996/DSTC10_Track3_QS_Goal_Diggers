import pickle
import sys
#with open('img_feature.bin','rb') as f:
with open(sys.argv[1],'rb') as f:
    d = pickle.load(f)

def get_one_item(index,d):
    dialogue_final,scene_thisround,id_final,type_final,bbox_final,label_final,image_feature = d[index]
    texta_list = dialogue_final[:-1] 
    texta = 'all '
    for text in texta_list:
         texta = texta + text
   
    #print(texta)
    textb = ''
    textb_list = type_final
    #print(textb_list) 
    for text in textb_list:
         textb = textb + ' ' + text
           
    #print(textb)
    #print(image_feature)
    #print(image_feature.shape)
    print(label_final)
    return texta,textb,image_feature,label_final
get_one_item(0,d)
get_one_item(1,d)

print(d[0])
