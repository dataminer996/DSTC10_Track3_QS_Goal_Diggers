import json
import jsonpath
import torch
import numpy as np
import pickle
import os, sys
import argparse
import copy
import numpy as np

def from_system_bin(dir_json, dir_binfile, dir_output):
    def getNonRepeatList(data):
        new_data = []
        for i in range(len(data)):
            if data[i] not in new_data:
                new_data.append(data[i])
        return new_data
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
    def get_json_value(json_data, key_name):
        key_value = jsonpath.jsonpath(json_data, '$..{key_name}'.format(key_name=key_name))
        return key_value

    with open(str(dir_binfile), 'rb') as f:
        binfile = pickle.load(f)
    with open(dir_json, 'rb') as j:
        jsonfile = json.load(j)

    # 1.打开bin文件
    # 2.获取did/拆解did
    # 3.生成json文件的from_system列表
    # 4.生成所有的s_object列表
    # 5.根据did生成的round和track在from_system列表中查找，如果是1，则获取s_object
    # 6.根据s_object对应bin文件中的all_object获取需要的下标
    # 7.生成新的objectid以及对应的bbox..列表
    # 8.替换bin文件中的内容
    # 9.dump出新的文件

    # 生成所有的s_object/from_system（通过列表就可以访问）
    dialogue = get_json_value(jsonfile, 'dialogue')
    dialogue_data = get_json_value(jsonfile, 'dialogue_data')[0]
    count = list(range(len(dialogue)))
    s_object_all = []
    from_system_all = []
    for i in count:
        # 统计这一轮所有的对话数量
        dialogue = get_dict_value(dialogue_data[i], 'dialogue', None)
        length_thisround = np.arange(len(dialogue))
        s_object_thisround = []
        from_system_thisround = []
        for j in length_thisround:
            s_object_thistrack = get_json_value(dialogue[j], 's_objects')
            s_object_thisround.append(s_object_thistrack[0])
            from_system_thistrack = get_json_value(dialogue[j], 'from_system')
            from_system_thisround.append(from_system_thistrack[0])
        s_object_thisround.append([s_object_thistrack])
        from_system_thisround.append(from_system_thistrack)
        s_object_all.append(s_object_thisround)
        from_system_all.append(from_system_thisround)


    bin_length = list(range(len(binfile)))
    for a in bin_length:
        binfile[a] = list(binfile[a])

        did = binfile[a][2]
        round = int(did / 100)
        track = did % 100

        objectid = binfile[a][3]
        type_all = binfile[a][4]
        bbox = binfile[a][5]
        label = binfile[a][6]
        image_feature = binfile[a][7]

        #if did != 166301:
        #     continue
        #print('object_id',objectid)
        if from_system_all[round][track] == str(1):
            type_new = []
            bbox_new = []
            label_new = []
            id_new = []
            feature_new = []
            feature_new.append(image_feature[0])
            object_new = s_object_all[round][track]
            #print("sobjectid",object_new)
            objectid_length = list(range(len(objectid)))
            findindexs = []
            #print("objectid_length",objectid_length)
            for b in objectid_length:
                if objectid[b] in object_new:
                    findindexs.append(b)
            #print("objectid_length remove",findindexs)
           
            for c in findindexs:
                type_new.append(type_all[c])
                bbox_new.append(bbox[c])
                label_new.append(label[c])
                id_new.append(objectid[c])
                feature_new.append(image_feature[c+1])

            binfile[a][3] = np.array(id_new)
            binfile[a][4] = np.array(type_new)
            binfile[a][5] = np.array(bbox_new)
            binfile[a][6] = np.array(label_new)
            binfile[a][7] = np.array(feature_new)
        #print('id_new',id_new)
        binfile[a] = tuple(binfile[a])


    with open(dir_output, 'wb') as save:
        pickle.dump(binfile, save)

    print('更换完毕')

dir_json = sys.argv[1]
dir_binfile = sys.argv[2]
dir_output = sys.argv[3]

from_system_bin(dir_json, dir_binfile, dir_output)
