import os
import glob
from util import utils
import json
import jsonpath
import numpy as np
import pickle
import argparse


def get_label_mapping(slot_mapping_path, action_mapping_path):
    slot_mapping = utils.load_pickle(slot_mapping_path)
    mapping_slot = {}
    for key, value in slot_mapping.items():
        mapping_slot[value] = key

    action_mapping = utils.load_pickle(action_mapping_path)
    mapping_action = {}
    for key, value in action_mapping.items():
        mapping_action[value] = key
    return slot_mapping, mapping_slot, action_mapping, mapping_action


def step3_process(step3_target_data, step3_pred_data):
    result = {}
    for index in range(len(step3_target_data)):
        line = step3_pred_data[index]
        line_json = json.loads(line.strip())
        pred_eid = str(line_json['eid'])
        label_predict = int(line_json['labels'])
        label_prob_list = line_json['label_prob'].strip('[').strip(']').split()
        label_prob = max([float(l) for l in label_prob_list])
        #print(label_predict)
        if label_predict == 1:
            target_list = step3_target_data[index].strip().split('\t')
            #print(target_list)
            eid = str(target_list[6])
            slot_key = target_list[3]
            slot_value = target_list[4]
            #print(slot_key, slot_value)
            #print(eid, pred_eid)
            assert eid == pred_eid
            if eid in result.keys():
                if slot_key in result[eid].keys():
                    result[eid][slot_key].append((slot_value, label_prob))
                else:
                    result[eid][slot_key] = [(slot_value, label_prob)]
            else:
                result[eid] = {slot_key: [(slot_value, label_prob)]}
        else:
            # result[eid] = {slot_key: [(slot_value, label_prob)]}
            continue
    print(result)
    output = {}
    for eid, value in result.items():
        output[eid] = {}
        for slot_key, sub_list in value.items():
            sub_list.sort(key=lambda x: x[1], reverse=True)
            output[eid][slot_key] = sub_list
    print(output)
    return output


def step3_embedding(step3_target_data, file_dir):
    pass


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
            print(round_num, track)
            data['dialogue_data'][round_num]['dialogue'][track]['transcript_annotated']['act_attributes']['objects'].append(object_fill)
        # with open('./predict_result/empty_user_obj.json', 'w') as f:
        #     json.dump(empty, f)

        return data

    data = produce_json_test(json_empty, binfile)
    return data


def parse_prediction(args):
    slot_mapping, mapping_slot, action_mapping, mapping_action = get_label_mapping(args['slot_mapping_path'],
                                                                                   args['action_mapping_path'])
    step1_pred_data = open(args['step1_pred_txt'], 'r').readlines()
    dir_bin = args['step2_bin']
    step3_pred_data = open(args['step3_pred_txt'], 'r').readlines()
    step3_target_data = open(args['step3_traget_txt'], 'r').readlines()

    #step3_pred_dict = step3_process(step3_target_data, step3_pred_data)
    #print(step3_pred_dict)
    output_result = {}
    for index, line in enumerate(step1_pred_data):
        if line:
            pred_json = json.loads(line)
            pred_eid = str(pred_json["eid"])
            pred_action = pred_json["action"]
            pred_action_prob = pred_json["action_prob"]
            pred_disambiguate = pred_json["disambiguate"]
            pred_disambiguate_prob = pred_json["disambiguate_prob"]
            pred_slot = pred_json["slot"]
            pred_slot_prob = pred_json["slot_prob"]

            pred_action_name = mapping_action[int(pred_action)]

            # assert str(eid) == pred_eid
            output_result[pred_eid] = {'act': pred_action_name,
                                       'act_attributes': {
                                           'slot_values': {},
                                           'request_slots': [],
                                           'objects': []
                                       }
                                       }
            #try:
            # for slot_key in step3_pred_dict[pred_eid].keys():
            #        if slot_key == 'request_slots':
            #            for slot_value, slot_prob in step3_pred_dict[pred_eid][slot_key]:
            #                if slot_prob > 0.5:
             #                   output_result[pred_eid]['act_attributes']['request_slots'].append(slot_value)
             #       elif slot_key == 'assetType':
             #           output_result[pred_eid]['act_attributes']['slot_values']['type'] = \
             #               step3_pred_dict[pred_eid][slot_key][0][0]
             #       elif slot_key == 'size':
             #           output_result[pred_eid]['act_attributes']['slot_values']['type'] = \
             #               step3_pred_dict[pred_eid][slot_key][0][0].upper()
             #       else:
             #           #print(step3_pred_dict[pred_eid][slot_key])
             #           output_result[pred_eid]['act_attributes']['slot_values'][slot_key] = \
             #               step3_pred_dict[pred_eid][slot_key][0][0]
            #except:
            #    continue
    output_result = step2_process(dir_bin, output_result)
    return output_result


def search_request_slot(request_slot, object_name, metadata_json):
    slot_dict = {}
    for objects in object_name:
        for slot in request_slot:
            slot_dict[slot] = metadata_json[objects][slot]
    return slot_dict


def get_objects_name(objects_list, scene_json):
    objects_name = []
    for objects_id in objects_list:
        for each in scene_json["scenes"][0]["objects"]:
            if each["index"] == objects_id:
                objects_name.append(each["prefab_path"])
    return objects_name


def get_scene(scene_id, turn_id):
    pic_keys = list(scene_id.keys())
    pic_values = list(scene_id.values())
    if (len(pic_keys)) >= 2:
        if turn_id >= int(pic_keys[1]):
            return pic_values[1]
        else:
            return pic_values[0]
    else:
        return pic_values[0]


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
        "--step1-pred-txt", help="Path to prediction of step1"
    )
    parser.add_argument(
        "--step2-bin", help="Path to prediction of step2"
    )
    parser.add_argument(
        "--step3-pred-txt", help="Path to prediction of step3"
    )
    parser.add_argument(
        "--step3-traget-txt", help="Path to target of step3"
    )
    parser.add_argument(
        "--slot-mapping-path", help="File of slot mapping"
    )
    parser.add_argument(
        "--action-mapping-path", help="File of action mapping"
    )
    parser.add_argument(
        "--metadata-path",
        required=True,
        help="Path of metadata",
    )
    parser.add_argument(
        "--scene-path",
        required=True,
        help="Path of scene json",
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
