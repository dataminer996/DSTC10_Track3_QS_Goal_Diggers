import json
import jsonpath
import torch
import numpy as np
import pickle
import os, sys
import argparse
import copy


# 生成所有devtest中的对应的objectid集合
dir_json = 'data/simmc2_dials_dstc10_devtest.json'
dir_bin = 'D:/data/simmc2/result.bin'

# 总函数
def final(dir_json, dir_bin, threshold):

    # 读取bin文件
    file = open(str(dir_bin), 'rb')
    binfile = pickle.load(file)
    # 读取json
    train = open(str(dir_json), 'r')
    data = json.load(train)

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

    def new_json(dir_json):
        train = open(str(dir_json), 'r')
        data = json.load(train)
        dialogue = (get_json_value(data, 'dialogue'))
        dialogue_data = get_json_value(data, 'dialogue_data')[0]
        count = list(range(len(dialogue)))
        for x in count:
            length_thisround = list(range(len(dialogue[x])))
            dia_thisround = dialogue[x]
            for y in length_thisround:
                change = dia_thisround[y]
                change['transcript_annotated']['act_attributes']['objects'] = []
        return data

    # 生成所有的devtest所有的objectid ([i,i,i,i,i,[j,j,j,j,j,j]])
    def object_search(dir_json):
        dialogue = (get_json_value(data, 'dialogue'))
        dialogue_data = get_json_value(data, 'dialogue_data')[0]
        sceneid = get_json_value(data, 'scene_ids')
        count = np.arange(len(dialogue))

        all_object = [] #生成所有的
        for i in count:

            dialogue = get_dict_value(dialogue_data[i], 'dialogue', None)
            length_thisround = np.arange(len(dialogue))

            # 统计这一轮所有对应的scenefile(即对应的图片文件名称，用列表保存，需要时按下标访问即可)
            scenefile = []
            keys = list(sceneid[i].keys())
            keys = [int(x) for x in keys]
            for r in length_thisround:
                if len(keys) == 2:
                    target_keys = keys[1]
                    if (r >= int(target_keys)) == True:
                        scenefile.append(sceneid[i][str(target_keys)])
                    else:
                        scenefile.append(sceneid[i]['0'])
                else:
                    scenefile.append(sceneid[i]['0'])
            all_id = []

            for j in length_thisround:
                scene_thisround = scenefile[j]
                dialogue_thisround = dialogue[j]
                scene = "./data/public/" + str(scene_thisround) + '_scene.json'
                if os.path.exists(scene):
                    pass
                else:
                    print(scene)
                with open(str(scene), 'r', encoding='utf-8-sig', errors='ignore') as f:
                    metadatafile = json.load(f, strict=False)
                id_thisround = get_json_value(metadatafile, 'index')
                all_id.append(id_thisround)
            all_object.append(all_id)
        return all_object
    all_object = object_search(dir_json) # [i[j[id],[id],[id]]]

    def produce_json_test(binfile, threshold):
        # 1.直接从路径中读取devempty.json
        empty = new_json(dir_json)
        dialogue = (get_json_value(empty, 'dialogue'))
        dialogue_data = get_json_value(empty, 'dialogue_data')[0]
        count = list(range(len(dialogue)))
        # 2.根据label概率生成需要填写的list的下标
        len_binfile = list(range(len(binfile)))
        need_fill = []
        for a in len_binfile:
            if float(binfile[a][3][1]) > threshold:
                need_fill.append(a)

        # 3.根据need_fill找到下标，获得did，并通过did进入轮数
        for b in need_fill:
            did = binfile[b][0]

            round = int(did/(1000*100))
            track = int((did/1000)%100)
            label = int(did%1000)

            # 获得需要添加的object
            object_fill = all_object[round][track][label]

            for c in count:
                dialogue = get_dict_value(dialogue_data[c], 'dialogue', None)
                length_thisround = np.arange(len(dialogue))
                if c == track:
                    for d in length_thisround:
                        if d == track:
                            empty['dialogue_data'][round]['dialogue'][track]['transcript_annotated']['act_attributes']['objects'].append(object_fill)
        return empty

    predict = produce_json_test(binfile, threshold)

    def result(target, predict):
        def evaluate_from_json(d_true, d_pred):
            """
            <list>d_true and <list>d_pred are in the following format:
            (Equivalent to "dialogue_data" field in the input data JSON file)
            [
                {
                    "dialogue": [
                        {
                            "belief_state": [
                                [
                                    {
                                        'act': <str>,
                                        'slots': [
                                            [
                                                SLOT_NAME, SLOT_VALUE
                                            ], ...
                                        ]
                                    },
                                    [End of a frame]
                                    ...
                                ],
                            ]
                        }
                        [End of a turn]
                        ...
                    ],
                }
                [End of a dialogue]
                ...
            ]
            """
            d_true_flattened = []
            d_pred_flattened = []

            for i in range(len(d_true)):
                # Iterate through each dialog
                dialog_true = d_true[i]["dialogue"]
                dialog_pred = d_pred[i]["dialogue"]
                dialogue_idx = d_true[i]["dialogue_idx"]

                for j in range(len(dialog_true)):
                    # Iterate through each turn
                    # turn_true = dialog_true[j]["belief_state"]
                    turn_true = dialog_true[j]["transcript_annotated"]
                    # turn_pred = dialog_pred[j]["belief_state"]
                    turn_pred = dialog_pred[j]["transcript_annotated"]

                    turn_true["turn_idx"] = j
                    turn_true["dialogue_idx"] = dialogue_idx

                    d_true_flattened.append(turn_true)
                    d_pred_flattened.append(turn_pred)

            return evaluate_from_flat_list(d_true_flattened, d_pred_flattened)

        def evaluate_from_flat_list(d_true, d_pred):
            """
            <list>d_true and <list>d_pred are in the following format:
            (Each element represents a single turn, with (multiple) frames)
            [
                [
                    {
                        'act': <str>,
                        'slots': [
                            [
                                SLOT_NAME, SLOT_VALUE
                            ], ...
                        ]
                    },
                    [End of a frame]
                    ...
                ],
                [End of a turn]
                ...
            ]
            """
            c = initialize_count_dict()

            # Count # corrects & # wrongs
            for i in range(len(d_true)):
                true_turn = d_true[i]
                pred_turn = d_pred[i]
                turn_evaluation = evaluate_turn(true_turn, pred_turn)
                # print(turn_evaluation)
                c = add_dicts(c, turn_evaluation)
            # print(c)
            # Calculate metrics
            joint_accuracy = c["n_correct_beliefs"] / c["n_frames"]

            act_rec, act_prec, act_f1 = rec_prec_f1(
                n_correct=c["n_correct_acts"], n_true=c["n_true_acts"], n_pred=c["n_pred_acts"]
            )

            slot_rec, slot_prec, slot_f1 = rec_prec_f1(
                n_correct=c["n_correct_slots"],
                n_true=c["n_true_slots"],
                n_pred=c["n_pred_slots"],
            )

            request_slot_rec, request_slot_prec, request_slot_f1 = rec_prec_f1(
                n_correct=c["n_correct_request_slots"],
                n_true=c["n_true_request_slots"],
                n_pred=c["n_pred_request_slots"],
            )

            object_rec, object_prec, object_f1 = rec_prec_f1(
                n_correct=c["n_correct_objects"],
                n_true=c["n_true_objects"],
                n_pred=c["n_pred_objects"],
            )

            # Calculate std err
            # print(c["n_true_acts"], c["n_pred_acts"], c["n_correct_acts"])
            # print(c["n_true_slots"], c["n_pred_slots"], c["n_correct_slots"])
            act_f1_stderr = d_f1(c["n_true_acts"], c["n_pred_acts"], c["n_correct_acts"])
            slot_f1_stderr = d_f1(c["n_true_slots"], c["n_pred_slots"], c["n_correct_slots"])
            request_slot_f1_stderr = d_f1(
                c["n_true_request_slots"],
                c["n_pred_request_slots"],
                c["n_correct_request_slots"],
            )
            object_f1_stderr = d_f1(
                c["n_true_objects"], c["n_pred_objects"], c["n_correct_objects"]
            )

            return {
                # "joint_accuracy": joint_accuracy,
                # "act_rec": act_rec,
                # "act_prec": act_prec,
                # "act_f1": act_f1,
                # "act_f1_stderr": act_f1_stderr,
                # "slot_rec": slot_rec,
                # "slot_prec": slot_prec,
                # "slot_f1": slot_f1,
                # "slot_f1_stderr": slot_f1_stderr,
                # "request_slot_rec": request_slot_rec,
                # "request_slot_prec": request_slot_prec,
                # "request_slot_f1": request_slot_f1,
                # "request_slot_f1_stderr": request_slot_f1_stderr,
                "object_rec": object_rec,
                "object_prec": object_prec,
                "object_f1": object_f1,
                "object_f1_stderr": object_f1_stderr,
            }

        def evaluate_turn(true_frame, pred_frame):
            count_dict = initialize_count_dict()

            # Must preserve order in which frames appear.
            # for frame_idx in range(len(true_turn)):
            # For each frame
            #    true_frame = true_turn[frame_idx]
            #    if frame_idx >= len(pred_turn):
            #        pred_frame = {}
            #    else:
            #        pred_frame = pred_turn[frame_idx]

            count_dict = add_dicts(
                count_dict, evaluate_frame(true_frame, pred_frame, strict=False)
            )

            return count_dict

        def evaluate_frame(true_frame, pred_frame, strict=True):
            """
            If strict=True,
                For each dialog_act (frame), set(slot values) must match.
                If dialog_act is incorrect, its set(slot values) is considered wrong.
            """
            count_dict = initialize_count_dict()
            count_dict["n_frames"] += 1

            # Compare Dialog Actss
            true_act = true_frame["act"] if "act" in true_frame else None
            pred_act = pred_frame["act"] if "act" in pred_frame else None
            b_correct_act = true_act == pred_act
            count_dict["n_correct_acts"] += b_correct_act
            count_dict["n_true_acts"] += "act" in true_frame
            count_dict["n_pred_acts"] += "act" in pred_frame

            # Compare Slots
            true_frame_slot_values = {f"{k}={v}" for k, v in true_frame['act_attributes']['slot_values'].items()}
            pred_frame_slot_values = {f"{k}={v}" for k, v in pred_frame['act_attributes']['slot_values'].items()}
            # print(true_frame_slot_values)

            count_dict["n_true_slots"] += len(true_frame_slot_values)
            count_dict["n_pred_slots"] += len(pred_frame_slot_values)

            if strict and not b_correct_act:
                pass
            else:
                count_dict["n_correct_slots"] += len(
                    true_frame_slot_values.intersection(pred_frame_slot_values)
                )

            # if len(true_frame_slot_values.intersection(pred_frame_slot_values)) != len(pred_frame_slot_values):
            # print(true_frame_slot_values)
            # print(pred_frame_slot_values)
            # print(len(true_frame_slot_values.intersection(pred_frame_slot_values)) == len(pred_frame_slot_values))
            # print('--')

            # Compare Request slots
            true_frame_request_slot_values = {rs for rs in true_frame['act_attributes']['request_slots']}
            pred_frame_request_slot_values = {rs for rs in pred_frame['act_attributes']['request_slots']}
            # print(true_frame_request_slot_values)

            count_dict["n_true_request_slots"] += len(true_frame_request_slot_values)
            count_dict["n_pred_request_slots"] += len(pred_frame_request_slot_values)

            if strict and not b_correct_act:
                pass
            else:
                count_dict["n_correct_request_slots"] += len(
                    true_frame_request_slot_values.intersection(pred_frame_request_slot_values)
                )

            # Compare Objects
            true_frame_object_values = {
                object_id for object_id in true_frame['act_attributes']['objects']
            }
            pred_frame_object_values = {
                object_id for object_id in pred_frame['act_attributes']['objects']
            }
            # print(true_frame_object_values)

            count_dict["n_true_objects"] += len(true_frame_object_values)
            count_dict["n_pred_objects"] += len(pred_frame_object_values)

            if strict and not b_correct_act:
                pass
            else:
                count_dict["n_correct_objects"] += len(
                    true_frame_object_values.intersection(pred_frame_object_values)
                )

            # Joint
            count_dict["n_correct_beliefs"] += (
                    b_correct_act
                    and true_frame_slot_values == pred_frame_slot_values
                    and true_frame_request_slot_values == pred_frame_request_slot_values
                    and true_frame_object_values == pred_frame_object_values
            )

            return count_dict

        def add_dicts(d1, d2):
            return {k: d1[k] + d2[k] for k in d1}

        def rec_prec_f1(n_correct, n_true, n_pred):
            rec = n_correct / n_true if n_true != 0 else 0
            prec = n_correct / n_pred if n_pred != 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0

            return rec, prec, f1

        def d_f1(n_true, n_pred, n_correct):
            # 1/r + 1/p = 2/F1
            # dr / r^2 + dp / p^2 = 2dF1 /F1^2
            # dF1 = 1/2 F1^2 (dr/r^2 + dp/p^2)
            dr = b_stderr(n_true, n_correct)
            dp = b_stderr(n_pred, n_correct)

            r = n_correct / n_true
            p = n_correct / n_pred
            f1 = 2 * p * r / (p + r) if p + r != 0 else 0

            d_f1 = 0.5 * f1 ** 2 * (dr / r ** 2 + dp / p ** 2)
            return d_f1

        def b_stderr(n_total, n_pos):
            return np.std(b_arr(n_total, n_pos)) / np.sqrt(n_total)

        def b_arr(n_total, n_pos):
            out = np.zeros(int(n_total))
            out[: int(n_pos)] = 1.0
            return out

        def initialize_count_dict():
            c = {
                "n_frames": 0.0,
                "n_true_acts": 0.0,
                "n_pred_acts": 0.0,
                "n_correct_acts": 0.0,
                "n_true_slots": 0.0,
                "n_pred_slots": 0.0,
                "n_correct_slots": 0.0,
                "n_true_request_slots": 0.0,
                "n_pred_request_slots": 0.0,
                "n_correct_request_slots": 0.0,
                "n_true_objects": 0.0,
                "n_pred_objects": 0.0,
                "n_correct_objects": 0.0,
                "n_correct_beliefs": 0.0,
            }
            return copy.deepcopy(c)

        json_target = target
        json_predicted = predict

        # Evaluate
        report = evaluate_from_json(
            json_target["dialogue_data"], json_predicted["dialogue_data"]
        )
        return report

    return result(data, predict)

alldir = sys.argv[1]
allfiles = os.listdir(alldir)
modeldirs = []
for filename in allfiles:
    ret = filename.find(sys.argv[2])
    if ret >=0 :
         checkpointdirs = os.listdir( alldir + "/"  + filename)
         for checkpointdir in checkpointdirs:
              ret1 = checkpointdir.find("checkpoint")
              if ret1 >= 0:
                 pytorchfiles =  os.listdir(alldir +  "/" + filename + "/" + checkpointdir)
                 for pytorchfile in pytorchfiles:
                       if pytorchfile == 'pytorch_model.bin':
                          if os.path.exists(alldir + "/" + filename + "/" + checkpointdir + "/result.txt"):
                               modeldirs.append(alldir +  "/" + filename + "/" + checkpointdir+ '/result.txt')
                          if os.path.exists(alldir + "/" + filename + "/" + checkpointdir + "/resultfromsys.txt"):
                               modeldirs.append(alldir +  "/" + filename + "/" + checkpointdir + "/resultfromsys.txt")
               
for modeldir in modeldirs:
       with open(modeldir,'r') as tf:
            lines = tf.readlines()
            num = 0
            for line in lines:
                  line = line.strip() 
                  if len(line) < 9:
                       continue
                  num = num + 1
                  if num == 3:
                    print(line,modeldir)


