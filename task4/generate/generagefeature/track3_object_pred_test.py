import json
import pickle
import utils_f
import jsonpath
import numpy as np
import os
import sys
import codecs
json_name =  sys.argv[1]
json_name_sys =  sys.argv[4]

#json_name = 'simmc2_dials_dstc10_train.json'
dir_name = './data/'

obj_num_file = sys.argv[2]
with open(obj_num_file) as f:
        num_data = json.load(f)

from_sys_file = sys.argv[3]
with open(from_sys_file, 'r') as r:
    from_sys_label = json.load(r)


def read_obj_fuction():
        
        global obj_function_fashion,obj_function_furniture
        dir_fashion = './data/fashion_result.txt'
        dir_furniture = './data/furniture_result.txt'
    
        txt = codecs.open(dir_fashion, mode='r', encoding='utf-8')
        # 解析每一行
        did_all = []
        obj_id_all = []
        type_all = []


        for line in txt:
            line = line.strip()
            imagetemp,temptypeobj = line.split("_objid_")
            scence = imagetemp[4:]
            objid,objtype = temptypeobj.split(".jpg,")
            if objtype == "tank_top":
                objtype = "tank top"
            if objtype == "shirt__vest":
                objtype = "shirt vest"

            did_all.append(scence)
            type_all.append(objtype)
            obj_id_all.append(int(objid))

        obj_function_fashion = (did_all,type_all,obj_id_all)
        


        
        txt = codecs.open(dir_furniture, mode='r', encoding='utf-8')
        # 解析每一行
        did_all_1 = []
        obj_id_all_1 = []
        type_all_1 = []

        for line in txt:
            line = line.strip()
            imagetemp,temptypeobj = line.split("_objid_")
            scence = imagetemp[4:]
            objid,objtype = temptypeobj.split(".jpg,")
            did_all_1.append(scence)
            type_all_1.append(objtype)
            obj_id_all_1.append(int(objid))

        obj_function_furniture = (did_all_1,type_all_1,obj_id_all_1)


        return

def search(json_name, dir_name, startid, endid):
    global num_data

#——————————————————————————————函数定义————————————————————————————————————————————————————————————
    # 数组去重
    def getNonRepeatList(data):
        new_data = []
        for i in range(len(data)):
            if data[i] not in new_data:
                new_data.append(data[i])
        return new_data

    # 数组分割
    def split_list_by_n(list_collection, n):
        """
        将集合均分，每份n个元素
        :param list_collection:
        :param n:
        :return:返回的结果为评分后的每份可迭代对象
        """
        for i in range(0, len(list_collection), n):
            yield list_collection[i: i + n]

    # 定义查找json文件内部值的函数
    # 模糊搜索json内部文件value
    def get_json_value(json_data, key_name):
        key_value = jsonpath.jsonpath(json_data, '$..{key_name}'.format(key_name=key_name))
        return key_value

    # 字典中查找嵌套的key
    def get_dict_value(date, keys, default=None):
        # default=None，在key值不存在的情况下，返回None
        keys_list = keys.split('.')
        # 以“.”为间隔，将字符串分裂为多个字符串，其实字符串为字典的键，保存在列表keys_list里
        if isinstance(date, dict):
            # 如果传入的数据为字典
            dictionary = dict(date)
            # 初始化字典
            for i in keys_list:
                # 按照keys_list顺序循环键值
                try:
                    if dictionary.get(i) != None:
                        dict_values = dictionary.get(i)
                    # 如果键对应的值不为空，返回对应的值
                    elif dictionary.get(i) == None:
                        dict_values = dictionary.get(int(i))
                    # 如果键对应的值为空，将字符串型的键转换为整数型，返回对应的值
                except:
                    return default
                    # 如果字符串型的键转换整数型错误，返回None
                dictionary = dict_values
            return dictionary
        else:
            # 如果传入的数据为非字典
            try:
                dictionary = dict(eval(date))
                # 如果传入的字符串数据格式为字典格式，转字典类型，不然返回None
                if isinstance(dictionary, dict):
                    for i in keys_list:
                        # 按照keys_list顺序循环键值
                        try:
                            if dictionary.get(i) != None:
                                dict_values = dictionary.get(i)
                            # 如果键对应的值不为空，返回对应的值
                            elif dictionary.get(i) == None:
                                dict_values = dictionary.get(int(i))
                            # 如果键对应的值为空，将字符串型的键转换为整数型，返回对应的值
                        except:
                            return default
                            # 如果字符串型的键转换整数型错误，返回None
                        dictionary = dict_values
                    return dictionary
            except:
                return default

    def get_replace_flag(pathlist, bbox_list):
        with open('./obj_thred.json', 'r') as f:
            data = json.load(f)
        for path, bbox in zip(pathlist, bbox_list):
            path = path.replace('/', '_')
            h_w, thred = data[path]
            yield (bbox[3] - bbox[0] if h_w == 'w' else bbox[2] - bbox[1]) >= thred
        # a = get_replace_flag(['MensCollection_Prefabs_Rearranged_long_coat_12'], [(1, 1, 70, 80)])

    
        
    def did_obj_type(imagefile, obj_id, domain):
        global obj_function_fashion,obj_function_furniture

        if domain == 'furniture':
               did_all,type_all,obj_id_all = obj_function_furniture
        else:
               did_all,type_all,obj_id_all = obj_function_fashion
            
        
        for i in range(len(did_all)):

            if imagefile == did_all[i]  and  obj_id_all[i] == int(obj_id):
                  #print("find the obj")                  
                  return type_all[i]

            #else:
        return "None"

    def getNonRepeatList(data):
        new_data = []
        for i in range(len(data)):
            if data[i] not in new_data:
                new_data.append(data[i])
        return new_data

    def s_object_get(did,dialogue):
        global from_sys_label
        final = []
#        round = int(int(did)/100)
        track = int(did)%100
        length = range(len(dialogue))
        dialogue_thisround = dialogue
        if from_sys_label[str(did)] == 0:
             return []
        
        for j in list(range(len(dialogue_thisround))):
                    if track == j:
                        for k in list(range(track)):
                            
                            used_object = dialogue_thisround[k]['system_transcript_annotated']['act_attributes']['objects']
                            if used_object!=[]:
                                for l in used_object:
                                    final.append(l)
        final = getNonRepeatList(final)
        return final


    def read_thisturn(dialogue_thisround, scenefile,domain,dialogue,did):

        # 统计本轮有多少个dialogue对话字典
        count_dialogue = np.arange(len(dialogue_thisround))

        # 1.获取对话内容（先用户后系统）
        dialogue_final = []  # 需要return
        objectsid = []
        # 进入本轮对话字典
        for x in count_dialogue:
            dialogue_final.append(get_dict_value(dialogue_thisround[x], 'transcript', None))
            dialogue_final.append(get_dict_value(dialogue_thisround[x], 'system_transcript', None))
            
            objectsid.append(get_dict_value(dialogue_thisround[x], 'transcript_annotated.act_attributes.objects', None))
        # 2.获取图片名称(将计数数组翻转后取第0个数字为下标)
        scene_thisround = scenefile  # 需要return

        # 3.获取对应图片文件里所有的index
        # 3.1打开文件
        scene = str(dir_name)+"public/" + str(scene_thisround) + '_scene.json'
        with open(str(scene), 'r', encoding='utf-8-sig', errors='ignore') as f:
            metadatafile = json.load(f, strict=False)

        # 3.2获取index
        all_id = get_json_value(metadatafile, 'index')  # 需要return
        all_prefab = get_json_value(metadatafile, 'prefab_path')
        all_bbox = get_json_value(metadatafile, 'bbox')


        if 1:
            final_total = []
            # 4.获取index对应的type
            # 4.1给出all_id对应的all_prefab列表（指该index下对应的prefabpath）
            for a in all_bbox:
                third = a[0] + a[3]
                fourth = a[1] + a[2]
                a[2] = third
                a[3] = fourth

            # 4.2将id和prefab打包成字典
            id_prefab = list(zip(all_id, all_prefab, all_bbox))
            sorted_id_prefab = sorted(id_prefab, key=lambda tup: tup[0])

            id_final = []
            prefab_final = []
            bbox_final = []
            for y in sorted_id_prefab:
                id_final.append(y[0])
                prefab_final.append(y[1])
                bbox_final.append(y[2])

            # 4.3.2进入循环添加all_type
            prefab_len = np.arange(len(id_prefab))
            type_final_old = []
            for l in prefab_len:
                temp = get_json_value(fashion, str(prefab_final[l]))
                if temp != False:
                    type_temp = get_json_value(temp[0], 'type')
                    type_final_old.append(type_temp[0])
                if temp == False:
                    temp = get_json_value(furniture, str(prefab_final[l]))
                    type_temp = get_json_value(temp[0], 'type')
                    type_final_old.append(type_temp[0])



            # label(需要这一轮对话所检索出的objectid)
            objects = objectsid[len(objectsid) - 1]
            objects_new = list(set(objects))
            # 将检索出的objects与indexid（sorted）相比较，生成label
            # 获取一致的元素的下标
            label_locate = []
            for p in objects:
                for q in prefab_len:
                    if p == id_final[q]:
                        label_locate.append(q)

            # 生成新数组
            label_final = [0] * prefab_len
            for e in prefab_len:
                for f in label_locate:
                    if e == f:
                        label_final[e] = 1
            # 获取image——feature
            scene_thisround_old = scene_thisround  

            scene_thisround_new = scene_thisround[2:] + ".png"
            scene_thisround_1  = scene_thisround + ".png"
            part1 = 'data/simmc2_scene_images_dstc10_public_part1'
            part2 = 'data/simmc2_scene_images_dstc10_public_part2'
            if os.path.exists(part1+"/"+scene_thisround_1):
                 scene_thisround = part1+"/"+scene_thisround_1
            elif os.path.exists(part2+"/"+scene_thisround_1):
                 scene_thisround = part2+"/"+scene_thisround_1
            elif os.path.exists(part2+"/"+scene_thisround_new):
                 scene_thisround = part2+"/"+scene_thisround_new
            elif os.path.exists(part1+"/"+scene_thisround_new):
                 scene_thisround = part1+"/"+scene_thisround_new
            else:  
                utils_f.save_imagef(scene_thisround + ".bin")
                print("error can't find===========image file ",scene_thisround)
                exit(0)
                 
            print("===========image file ",scene_thisround)
            type_final = []
            for m in prefab_len:
                obj_id = id_final[m]
                domain = domain
                temp_new = did_obj_type(scene_thisround_old, obj_id, domain)
                if temp_new!=None:
                    temp_new = temp_new.strip()
                type_final.append(temp_new)

            if type_final != type_final_old:
              print("new type",type_final)
              print("old type",type_final_old)

            if os.path.basename(scene_thisround)  ==  'cloth_store_1416238_woman_4_8.png' or    os.path.basename(scene_thisround)  ==  'cloth_store_1416238_woman_19_0.png':
                 return  (-2,scene_thisround,id_final,type_final,bbox_final,label_final,None,prefab_final,0,0)




            bbox_length = list(range(len(bbox_final)))
            flag_list = []
            #for r in bbox_length:
            #    bool_bbox = get_replace_flag(list(prefab_final[r]), list(bbox_final[r]))
            #    flag_list.append(bool_bbox[0])
            #bool_bbox = get_replace_flag(prefab_final, bbox_final)
            #flag_list = list(bool_bbox)
            #print(flag_list)
            obj_num = 0
            for label in label_final:
                 if label == 1:
                    obj_num = obj_num + 1
            if (len(objects_new) != obj_num):
               print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++find the error objnum and objid", len(objects),obj_num,objects,id_final)
            if obj_num != 0 or len(objects) == 0:
               #image_feature = readimage_feature(scene_thisround, bbox_final)
               image_feature = utils_f.readimagefrombin_feature_old(scene_thisround, bbox_final)
               #image_feature = readimagefrombin_feature(scene_thisround, bbox_final)
            else:               
               return  (-1,scene_thisround,id_final,type_final,bbox_final,label_final,None,prefab_final,0,0)
            #image_feature = utils_f.readimagefrombin_feature(scene_thisround, bbox_final,flag_list,prefab_final)
        #    image_feature = readimage_feature(scene_thisround, bbox_final)
            return (dialogue_final,scene_thisround,id_final,type_final,bbox_final,label_final,image_feature,prefab_final,len(objects_new),obj_num)


    train = open(str(json_name), 'r')
    data = json.load(train)


    train = open(str(json_name_sys), 'r')
    sysdata = json.load(train)

    # 伪函数，read_image
    def readimage_feature(imagename, bbox_list):
        image_feature = []
        for i in range(len(bbox_list)):
            a = np.zeros((2054))
            image_feature.append(a)

        return image_feature

    # ——————————————————————————————————————————————————————————————————————————————————————————————————————————

    # metadata打开
    meta1 = open(str(dir_name)+'fashion_prefab_metadata_all.json', 'r')
    fashion = json.load(meta1)
    meta2 = open(str(dir_name)+'furniture_prefab_metadata_all.json', 'r')
    furniture = json.load(meta2)

    # 全局通用
    dialogue = (get_json_value(data, 'dialogue'))
    dialogue_data = get_json_value(data, 'dialogue_data')[0]
    dialogue_data_sys = get_json_value(sysdata, 'dialogue_data')[0]

    sceneid = get_json_value(data, 'scene_ids')

    # 计数 用于标记dialogue_data下的对话
    # count = []
    # for index, value in enumerate(dialogue):
    #     if value:
    #         count.append(index)
    len_d =  len(dialogue_data)
    if endid > len_d:
          count = np.arange(startid, len_d)
    else:
          count = np.arange(startid, endid)
    count_num = 0 
    save = []
    picture_mistake = []
    picture_mistake_all =[]
    
    for i in count:
        print('进入对话第', i, '轮')  # 进入对话（例如对话第零轮）
        # 统计这一轮所有的对话数量
        dialogue = get_dict_value(dialogue_data[i], 'dialogue', None)
        dialogue_sys = get_dict_value(dialogue_data_sys[i], 'dialogue', None)

        domain = get_dict_value(dialogue_data[i], 'domain', None)

        length_thisround = np.arange(len(dialogue))

        # 统计这一轮所有对应的scenefile(即对应的图片文件名称，用列表保存，需要时按下标访问即可)
        scenefile = sceneid[i]


        # 建立一个大的list包含所有的
        for j in length_thisround:
        
            diag_id = i * 100 + j
            print('已经进入本轮中第', j, '个用例')
            scene_track = "-1"
            if num_data[str(diag_id)] == 0:
                   print("continue ",diag_id)
                   continue
            
            if len(list(scenefile.keys())) >= 2:
                if j >= int(list(scenefile.keys())[1]):
                    scene_round = list(scenefile.values())[1]
                    scene_track = list(scenefile.values())[0]

                else:
                    scene_round = list(scenefile.values())[0]                    
                    scene_track = list(scenefile.values())[1]
            else:
                scene_round = list(scenefile.values())[0]

            # 拆分对话，本轮对话为0,1,2,...j总和
            count_j = np.arange(j + 1)
            # print('本次用例涵盖的dialogue有', count_j)
            dialogue_thisround = []  # 这一次使用的具体用例
            for k in count_j:
                dialogue_thisround.append(dialogue[k])
            # print('本轮使用的dialogue是', dialogue_thisround)
            # 调用函数read_oneturn
            if 1:
              dialogue_final,scene_thisround,id_final,type_final,bbox_final,label_final,image_feature,prefab_final,objnum,findnum = read_thisturn(dialogue_thisround, scene_round,domain,dialogue_sys,diag_id)
              if dialogue_final == -2:
                   continue
              if dialogue_final != -1 and objnum > findnum:                  
                  save.append((dialogue_final,scene_thisround,diag_id,id_final,type_final,bbox_final,label_final,image_feature,prefab_final))
              
              if dialogue_final == -1 or objnum > findnum:
                  if scene_track != '-1':
                       dialogue_final,scene_thisround,id_final,type_final,bbox_final,label_final,image_feature,prefab_final,objnum1,findnum1  = read_thisturn(dialogue_thisround, scene_track,domain,dialogue_sys,diag_id)
                       if dialogue_final == -1:
                            print('+++++++++++++++find the error id', int(i*100+j),objnum,objnum1,findnum,findnum1)
                            picture_mistake_all.append(int(i * 100 + j))
                       elif objnum1 <= findnum + findnum1:
                            picture_mistake.append(int(i * 100 + j))
                       else:
                            picture_mistake_all.append(int(i * 100 + j))                            
                            print('+++++++++++++++find the error id', int(i*100+j),objnum,findnum,findnum1)
                  else:
                       picture_mistake_all.append(int(i * 100 + j))
                       print('+++++++++++++++find the error id', int(i*100+j),objnum,findnum)
                       continue
              
                  print('picture_mistake_did', int(i*100+j),objnum,findnum,findnum1)
           # except:            
           #     picture_mistake_all.append(int(i * 100 + j))
           #     print("++++++++++++++++++++++++++++find the error ===",diag_id)
           #     continue

           # if len(image_feature) < len(label_final):
           #         print("")
           #         continue  
            if dialogue_final != -1:
                if len(image_feature) != len(bbox_final) + 1:
                    print("++++++++++++++++++++++++++++find image feature error ===",i,len(image_feature),len(bbox_final))
                    #print("")
                    picture_mistake_all.append(int(i * 100 + j))
                    continue  
            #first_image_feature = [image_feature[0]]
            #print(len(image_feature[0]+image_feature[1:2]))
            #print((image_feature[0]+image_feature[1:2]))
            #num = int(len(label_final)/50)
          #  if len(label_final) > 50:
          #      for i in range(num):
          #          if i < len(label_final)/50:
          #               #try:
          #                  save.append((dialogue_final,scene_thisround,id_final[i*50:(i+1)*50],type_final[i*50:(i+1)*50], bbox_final[i*50:(i+1)*50], label_final[i*50:(i+1)*50], first_image_feature+image_feature[1+i*50:1+(i+1)*50]))
                         #except:
                         #   continue
          #      if len(label_final)%50 != 0:
          #             i = int(len(label_final) / 50)
          #             print(i*50)
          #             print(id_final[i*50:])
          #             save.append((dialogue_final,scene_thisround,id_final[i*50:],type_final[i*50:], bbox_final[i*50:], label_final[i*50:], first_image_feature+image_feature[1+i*50:]))
          #  else:
            save.append((dialogue_final,scene_thisround,diag_id,id_final,type_final,bbox_final,label_final,image_feature,prefab_final))
            count_num = count_num + 1
            if count_num % 1000 == 0:
                 print("image feature save")
                 utils_f.save_imagef(startid)
                 #CUDA_VISIBLE_DEVICES=4 python3 searchtotal_new.py 0 2500
    filename = "object_pred.bin"
    print("filename",filename)
    with open(filename, 'wb') as fp:
         pickle.dump(save,fp)
    with open('picture_mistake.bin', 'wb') as picture:
        pickle.dump(picture_mistake, picture)
    with open('picture_mistake_all.bin', 'wb') as picture:
        pickle.dump(picture_mistake_all, picture)

    print('完毕')
    
read_obj_fuction()

startid = 0

endid = 10000

#startid = int(sys.argv[1])
#endid = int(sys.argv[2])
utils_f.init_imagef("devtestobj")
search(json_name, dir_name, startid, endid)
#utils_f.save_imagef("0820obj"+str(startid))
