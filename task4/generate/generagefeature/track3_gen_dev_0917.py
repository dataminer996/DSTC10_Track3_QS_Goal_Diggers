import json
import pickle
import utils_f
import jsonpath
import numpy as np
import os
import sys
sys.path.append(os.getcwd())

json_name = 'simmc2_dials_dstc10_devtest.json'

#json_name = 'simmc2_dials_dstc10_train.json'
dir_name = './data/'


def search(json_name, dir_name, startid, endid):
    # ——————————————————————————————函数定义————————————————————————————————————————————————————————————
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

    # 伪函数，read_image
    def readimage_feature(imagename, bbox_list):
        image_feature = []
        for i in range(len(bbox_list)):
            a = np.zeros((2054))
            image_feature.append(a)

        return image_feature

    def read_thisturn(dialogue_thisround, scenefile):

        # 统计本轮有多少个dialogue对话字典
        count_dialogue = np.arange(len(dialogue_thisround))

        # 1.获取对话内容（先用户后系统）
        dialogue_final = []  # 需要return
        objectsid = []
        slot_values = []
        # 进入本轮对话字典
        for x in count_dialogue:
            dialogue_final.append(get_dict_value(dialogue_thisround[x], 'transcript', None))
            dialogue_final.append(get_dict_value(dialogue_thisround[x], 'system_transcript', None))
            objectsid.append(get_dict_value(dialogue_thisround[x], 'system_transcript_annotated.act_attributes.objects', None))
            slot_values.append(get_dict_value(dialogue_thisround[x], 'system_transcript_annotated.act_attributes.slot_values', None))

        # 2.获取图片名称(将计数数组翻转后取第0个数字为下标)
        scene_thisround = scenefile # 需要return

        # 3.获取对应图片文件里所有的index
        # 3.1打开文件
        scene = str(dir_name) + "public/" + str(scene_thisround) + '_scene.json'
        with open(str(scene), 'r', encoding='utf-8-sig', errors='ignore') as f:
            metadatafile = json.load(f, strict=False)

        # 3.2获取index
        all_id = get_json_value(metadatafile, 'index')  # 需要return
        all_prefab = get_json_value(metadatafile, 'prefab_path')
        all_bbox = get_json_value(metadatafile, 'bbox')
        prefab_len = np.arange(len(all_prefab))

        for a in all_bbox:
            third = a[0] + a[3]
            fourth = a[1] + a[2]
            a[2] = third
            a[3] = fourth

        #本轮用到的id
        part_id = objectsid[len(objectsid) - 1]
        part_value = slot_values[len(objectsid)-1]
        label_locate = []
        part_prefab = []
        part_bbox = []
        for p in part_id:
            for q in prefab_len:
                if p == all_id[q]:
                    label_locate.append(q)
                    part_prefab.append(all_prefab[q])
                    part_bbox.append(all_bbox[q])

        id_final = part_id
        prefab_final = part_prefab
        bbox_final = part_bbox
        slotvalue_final = []
        # slot_value 处理
        if part_value:
            Object_slot = list(part_value.values())
            if isinstance(Object_slot[0], dict):
                for o in Object_slot:
                    temp = []
                    print('字典内部值是',o)
                    for key, value in o.items():
                        text = ''
                        if isinstance(value, list):
                            for q in value:
                                text = text + str(q)+ ' '
                            n = str(key) + ':' + str(text)
                            slotvalue_final.append(n)
                        else:
                            n = str(key) + ':' + str(value)
                            slotvalue_final.append(n)
                slotvalue_final = getNonRepeatList(slotvalue_final)
            else:
                part_value = [part_value]
                for m in part_value:
                    for key, value in m.items():
                        n = str(key) + ':' + str(value)
                        slotvalue_final.append(n)
        else:
            part_value = []

        print("slotvalue_final", slotvalue_final)
        slotvalue_final_fix = []
        slotvalue_final_symbol = slotvalue_final
        if len(slotvalue_final) > 1:
            for g in slotvalue_final:
                print('slotvalue内部值分别是', g)
                if isinstance(g, list):
                    if len(g) >1:
                        print('可以修改', len(g))
                        value_fix = ','
                        # for h in g:
                        #     print('该内部值的值分别是', h)
                        #     value_fix += str(h)
                        # slotvalue_final_fix.append(value_fix)
                        value_fix = value_fix.join(g)
                        slotvalue_final_fix.append(value_fix)
                        slotvalue_final = slotvalue_final_fix
        if slotvalue_final != slotvalue_final_symbol:
            print('修改后的slotvaluefinal', slotvalue_final)
        else:
            print('未经修改')
        # 4.3.2进入循环添加all_type
        prefab_len = np.arange(len(prefab_final))
        type_final = []
        image_feature = []
        if id_final != []:
            for l in prefab_len:
                temp = get_json_value(fashion, str(prefab_final[l]))
                if temp != False:
                    type_temp = get_json_value(temp[0], 'type')
                    type_final.append(type_temp[0])
                if temp == False:
                    temp = get_json_value(furniture, str(prefab_final[l]))
                    type_temp = get_json_value(temp[0], 'type')
                    type_final.append(type_temp[0])


            # 获取image——feature
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

        #if label in label_final
        if len(objectsid) != len(bbox_final):
           return (-1,scene_thisround, id_final, type_final, bbox_final, slotvalue_final, None)
            
                        
        image_feature = utils_f.readimagefrombin_feature(scene_thisround, bbox_final)
        
        # slotvalue全部添加

        
		
		
		
		#dialogue_final.remove(dialogue_final[len(dialogue_final) - 1])
        #image_feature = readimage_feature(scene_thisround, bbox_final)
        return (dialogue_final, scene_thisround, id_final, type_final, bbox_final, slotvalue_final, image_feature)




    # ——————————————————————————————————————————————————函数部分结束————————————————————————————————————————————————————————


    train = open(str(dir_name) + str(json_name), 'r')
    data = json.load(train)


    # metadata打开
    meta1 = open(str(dir_name) + 'fashion_prefab_metadata_all.json', 'r')
    fashion = json.load(meta1)
    meta2 = open(str(dir_name) + 'furniture_prefab_metadata_all.json', 'r')
    furniture = json.load(meta2)

    # 全局通用
    dialogue = (get_json_value(data, 'dialogue'))
    dialogue_data = get_json_value(data, 'dialogue_data')[0]
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
        length_thisround = np.arange(len(dialogue))

        # 统计这一轮所有对应的scenefile(即对应的图片文件名称，用列表保存，需要时按下标访问即可)
        scenefile = sceneid[i]

        # 建立一个大的list包含所有的
        for j in length_thisround:
            print('已经进入本轮中第', j, '个用例')
            scene_track = "-1"
            if len(list(scenefile.keys())) >= 2:
                if j >= list(scenefile.keys())[1]:
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

            try:
              dialogue_final, scene_thisround, id_final, type_final, bbox_final, slotvalue_final, image_feature = read_thisturn(
                dialogue_thisround, scene_round)
              if dialogue_final == -1:
                  if scene_track != '-1':
                       dialogue_final, scene_thisround, id_final, type_final, bbox_final, slotvalue_final, image_feature = read_thisturn(dialogue_thisround, scene_track)
                       if dialogue_final == -1:
                            picture_mistake_all.append(int(i * 100 + j))
                       else:
                            picture_mistake.append(int(i * 100 + j))
                  else:
                       picture_mistake_all.append(int(i * 100 + j))

                  print('picture_mistake_did', int(i*100+j))
            except:            
               picture_mistake_all.append(int(i * 100 + j))
               continue
            #
            # print('id_final\n', id_final)
            # print('slotvalue_final\n', slotvalue_final)

            # first_image_feature = [image_feature[0]]
            if image_feature != []:
                first_image_feature = [image_feature[0]]
            else:
                first_image_feature = image_feature
            # print(len(image_feature[0]+image_feature[1:2]))
            # print((image_feature[0]+image_feature[1:2]))
            dialog_id = i * 100 + j
            save.append(
                    (dialogue_final, scene_thisround,dialog_id, id_final, type_final, bbox_final, slotvalue_final, image_feature))
            count_num = count_num + 1
            if count_num % 100 == 0:
                print("image feature save")
                utils_f.save_imagef("gen"+ str(startid))
                # CUDA_VISIBLE_DEVICES=4 python3 searchtotal_new.py 0 2500
    filename = "0820v2gendevdata" +str(startid) + '.bin'
    with open(filename, 'wb') as fp:
        pickle.dump(save, fp)
    with open(filename + 'picture_mistake.bin', 'wb') as picture:
        pickle.dump(picture_mistake, picture)
    with open(filename + 'picture_mistake_all.bin', 'wb') as picture:
        pickle.dump(picture_mistake_all, picture)

    print('完毕')

startid = int(sys.argv[1])
endid = int(sys.argv[2])
utils_f.init_imagef("gendev0820v2"+ str(startid))
search(json_name, dir_name, startid, endid)
utils_f.save_imagef("gendev0820v2"+ str(startid))
