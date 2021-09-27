import json
import jsonpath
import pickle
import numpy as np
import pandas as pd
dir_input = './json/simmc2_dials_dstc10_devtest.json'
dir_output = './predict_result/train_s_object.json'

def s_object_produce(dir_input, dir_output):
    with open(dir_input, 'r') as r:
        data = json.load(r)
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
    def s_object_get(did):
        final = []
        round = int(int(did)/100)
        track = int(did)%100
        for i in length:
            if round == i:
                dialogue_thisround = dialogue[i]
                for j in list(range(len(dialogue_thisround))):
                    if track == j:
                        for k in list(range(track)):
                            used_object = dialogue_thisround[k]['system_transcript_annotated']['act_attributes']['objects']
                            if used_object!=[]:
                                for l in used_object:
                                    final.append(l)
        final = getNonRepeatList(final)
        return final

    dialogue = get_json_value(data, 'dialogue')
    length = list(range(len(dialogue)))

    for i in length:
        dialogue_thisround = dialogue[i]
        for j in list(range(len(dialogue_thisround))):
            did = int(i*100 + j)
            dialogue_thistrack = dialogue_thisround[j]
            user_objectid_now = dialogue_thistrack['transcript_annotated']['act_attributes']['objects']
            if user_objectid_now != []:
                user_objectid_now = np.array(dialogue_thistrack['transcript_annotated']['act_attributes']['objects'])
                sys_objectid_before = np.array(s_object_get(did))
                if np.any(np.in1d(sys_objectid_before, user_objectid_now)) == True:
                    s_objects = s_object_get(did)
                    dialogue_thistrack['transcript_annotated']['act_attributes']['from_system'] = 1
                    dialogue_thistrack['transcript_annotated']['act_attributes']['s_object'] = s_objects
            else:
                dialogue_thistrack['transcript_annotated']['act_attributes']['from_system'] = 0
                dialogue_thistrack['transcript_annotated']['act_attributes']['s_object'] = []
    with open(dir_output, 'w') as w:
        json.dump(data, w)

s_object_produce(dir_input, dir_output)