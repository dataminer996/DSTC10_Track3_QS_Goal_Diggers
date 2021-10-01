import os
import glob
from util import utils
import json
import jsonpath
import numpy as np
import pickle
import argparse


def step2_process(dir_bin, json_empty):

    # 读取bin文件
    file = open(str(dir_bin), 'rb')
    binfile = pickle.load(file)
    # 读取json
    #empty = open(str(dir_empty), 'r')
    #json_empty = json.load(empty)

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

    def produce_json_test(data, binfile):
        #dialogue = (get_json_value(empty, 'dialogue'))
        #dialogue_data = get_json_value(empty, 'dialogue_data')
        #print(dialogue_data)
        #count = list(range(len(dialogue_data)))
        # 2.根据label概率生成需要填写的list的下标
        len_binfile = list(range(len(binfile)))
        need_fill = []
        for a in len_binfile:
            #print(binfile[a])
            if float(binfile[a][5]) == 1.0:
                need_fill.append(a)
        # 3.根据need_fill找到下标，获得did，并通过did进入轮数
        for b in need_fill:
            did = binfile[b][0]

            round_num = int(did / 100)
            track = int(did % 100)
            #label = int(did % 1000)
            # # 获得需要添加的object
            # object_fill = all_object[round][track][label]
            object_fill = int(binfile[b][4])
            #for c in count:
            #    dialogue = get_dict_value(dialogue_data[c], 'dialogue', None)
            #    length_thisround = np.arange(len(dialogue))
            #    if c == track:
            #        for d in length_thisround:
            #            if d == track:
            #print(round_num, track)
            data['dialogue_data'][round_num]['dialogue'][track]['transcript_annotated']['act_attributes']['objects'].append(object_fill)
        # with open('./predict_result/empty_user_obj.json', 'w') as f:
        #     json.dump(empty, f)

        return data

    data = produce_json_test(json_empty, binfile)
    return data


def main(args):
    dialogs = json.load(open(args['split_path'], 'r'))
    
    output = open(args['save_path'], 'w+')
    dialogs_output = {"dialogue_data": []}

    for dialog_id, dialog_datum in enumerate(dialogs["dialogue_data"]):
        dialogue_idx = dialog_datum["dialogue_idx"]
        domain = dialog_datum["domain"]
        scene_ids = dialog_datum["scene_ids"]
        tmp = {"dialogue_idx": dialogue_idx, "dialogue": []}
        for turn_id, turn_datum in enumerate(dialog_datum["dialogue"]):
            tmp["dialogue"].append({'turn_idx': turn_datum['turn_idx'],
                                    'transcript_annotated': {'act': '',
                                       'act_attributes': {
                                           'slot_values': {},
                                           'request_slots': [],
                                           'objects': []
                                       }
                                       }})
        dialogs_output["dialogue_data"].append(tmp)
    dialogs_output = step2_process(args['step2_bin'], dialogs_output)
    json.dump(dialogs_output, output)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step2-bin", help="Path to prediction of step2"
    )
    parser.add_argument(
        "--split-path", help="Process SIMMC file of test phase"
    )
    parser.add_argument(
        "--save-path",
        required=True,
        help="Path to save SIMMC dataset",
    )

    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
