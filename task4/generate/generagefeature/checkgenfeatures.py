import pickle
import random
import sys
import numpy as np
filename = sys.argv[1]
with open(filename,'rb') as f:
     features = pickle.load(f)


def get_one_item(index,features):
        print("===========================================================",index)
       # if index == 11:
       #     print(features[index]) 
        dialogue_final, scene_thisround,dialog_id, id_final, type_final, bbox_final, slotvalue_final, image_feature = features[index]

      
        textc_list = dialogue_final[:-1]
        textc = ''
        for text in textc_list:
             textc = textc + text

        caption_list = dialogue_final[-1:]
        caption = ''
        for text in caption_list:
             caption = caption + text


        textd_list = slotvalue_final
        if len(textd_list) <= 1:
            textd = ''
        else:
            textd = "first:"
            
        num = 0
        #print(textb_list) 
        for text in textd_list:
             num = num + 1
             if textd == '':
                 textd = text
             else:
                 if num == 1:
         #            if index == 11:
        #                print(text[0])
          #           print("type",type(text))
                     textd = "first:"  + text                    
                 if num == 2:
                    textd = textd + ' second:' + text
                 if num == 3:
                    textd = textd + ' third:' + text
                 if num == 4:
                     textd = textd + ' fourth:' + text
                 if num == 5:
                      textd = textd + ' fifth:' + text
                 if num == 6:
                      textd = textd + ' sixth:' + text
                 if num == 7:
                      textd = textd + ' seventh:' + text
                 if num == 8:
                      textd = textd + ' eighth:' + text
                 if num == 9:
                      textd = textd + ' ninth:' + text
                 if num == 10:
                      textd = textd + ' tenth:' + text                     

        #print(texta)
        textb = ''
        textb_list = type_final
        #print(textb_list) 
        for text in textb_list:
             if textb == '':
                 textb = text
             else:
                 textb = textb + ' ' + text

        #print(textb)
      #  print(image_feature)
        image_feature = np.array(image_feature)
       # print(image_feature.shape)
        #print(label_final)
        return caption,str(textb),str(textc),str(textd),image_feature


for i in range(len(features )):
    ret = get_one_item(i,features)
    print(ret)


