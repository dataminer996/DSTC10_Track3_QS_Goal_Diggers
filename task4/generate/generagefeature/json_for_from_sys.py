import json
import jsonpath
import pickle
import numpy as np
import pandas as pd
import sys

dir_devtest = sys.argv[1]
dir_label = sys.argv[2]
dir_output = 'test_s_object_pred.json'
def jsonfile_producr(dir_devtest, dir_label):
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
    with open(dir_devtest, 'r') as f:
        data = json.load(f)

    with open(dir_label, 'r') as r:
        label_json = json.load(r)
    dialogue = get_json_value(data, 'dialogue')
    dialogue_id = get_json_value(data, 'dialogue_idx')
    length = list(range(len(dialogue)))

    def s_object_get(did,dialogue):
        final = []
        roundid = int(int(did)/100)
        track = int(did)%100
        #for i in length:
         #   if round == i:
        i = roundid
        dialogue_thisround = dialogue[roundid]
        for j in list(range(len(dialogue_thisround))):
                    if track == j:
                        for k in list(range(track)):
                            used_object = dialogue_thisround[k]['system_transcript_annotated']['act_attributes']['objects']
                            if used_object!=[]:
                                for l in used_object:
                                    final.append(l)
        final = getNonRepeatList(final)
        return final

    for i in length:
        dialogue_thisround = dialogue[i]
        for j in list(range(len(dialogue_thisround))):
            dialogue_thistrack = dialogue_thisround[j]
            did = str(i*100 + j)
            if label_json[did] == 1:
                dialogue_thistrack['transcript_annotated']['act_attributes']['from_system'] = "1"
                s_object = s_object_get(did,dialogue)
                dialogue_thistrack['transcript_annotated']['act_attributes']['s_objects'] = s_object
            elif label_json[did] == 0 or label_json[did] == 2 or label_json[did] == -1:
                dialogue_thistrack['transcript_annotated']['act_attributes']['from_system'] = "0"
                dialogue_thistrack['transcript_annotated']['act_attributes']['s_objects'] = []
            else:
                print('出现异常')
    with open(dir_output, 'w') as w:
        json.dump(data, w)
    return data
jsonfile_producr(dir_devtest, dir_label)
