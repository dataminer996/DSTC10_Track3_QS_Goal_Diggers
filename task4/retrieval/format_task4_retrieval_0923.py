import json
import jsonpath
import pickle
import argparse
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

def main(args):
    with open(args['split_path'], 'r') as r:
        data = json.load(r)
    with open(args['dir_bin'], 'rb') as f:
        binfile = pickle.load(f)
    import copy 
    write_data = []
    for item in data.get('retrieval_candidates'):
        component = dict()
        component['dialog_id'] = item.get('dialogue_idx')
        component['candidate_scores'] = copy.deepcopy(item.get('retrieval_candidates'))
        for subitem in component.get('candidate_scores'):
            subitem['scores'] = subitem.get('retrieval_candidates')
            del subitem['retrieval_candidates']
            del subitem['gt_index']
        write_data.append(component)

    for item in binfile:
        did = item[0].item()
        _, logis = item[-2].tolist()
        dial = item[-1].item()
        big = did // 100
        small = did % 100
        index = data.get('retrieval_candidates')[big].get('retrieval_candidates')[small].get(
            'retrieval_candidates').index(dial)
        write_data[big].get('candidate_scores')[small].get('scores')[index] = logis

    with open(args['save_path'],'w') as write:
        #json.dump(data_new, write)
        json.dump(write_data, write)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir-bin", help="Path dir to bin"
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
        main(parsed_args)
    except (IOError) as msg:
        parser.error(str(msg))
