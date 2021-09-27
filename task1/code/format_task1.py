import os
import json
import glob
import argparse


def parse_prediction(file_name):
    step1_pred_data = open(file_name, 'r').readlines()

    output_result = {}
    system_use = {}
    for index, line in enumerate(step1_pred_data):
        if line:
            pred_json = json.loads(line)
            pred_eid = str(pred_json["eid"])
            pred_disambiguate = int(pred_json["disambiguate"])
            pred_disambiguate_prob = max([float(each) for each in pred_json["disambiguate_prob"]])
            output_result[pred_eid] = [pred_disambiguate, pred_disambiguate_prob]

            pred_from_system_prob = max([float(each) for each in pred_json["from_system_prob"]])
            pred_objects_num_prob = max([float(each) for each in pred_json["objects_num_prob"]])
            pred_action_prob = max([float(each) for each in pred_json["action_prob"]])
            pred_from_system = [int(pred_json["from_system"]), pred_from_system_prob]
            pred_objects_num = [int(pred_json["objects_num"]), pred_objects_num_prob]
            pred_action = [int(pred_json["action"]), pred_action_prob]
            system_use[pred_eid] = {"from_system": pred_from_system, "objects_num": pred_objects_num, 'action': pred_action}

    return output_result, system_use


def embedding_prediction(file_dir):
    if os.path.isdir(file_dir):
        file_list = glob.glob(file_dir + '/*')
    else:
        file_list = [file_dir]
    merge_dict = {}
    merge_from_system = {}
    merge_objects_num = {}
    merge_action = {}
    for file_name in file_list:
        parse_result, system_use = parse_prediction(file_name)
        for eid, label in parse_result.items():
            if eid not in merge_dict.keys():
                merge_dict[eid] = {1: 0.0, 0: 0.0}
            merge_dict[eid][label[0]] += label[1]

        for eid, label in system_use.items():
            from_system_num = label["from_system"][0]
            objects_num = label["objects_num"][0]
            action = label["action"][0]

            if eid not in merge_from_system.keys():
                merge_from_system[eid] = {from_system_num: {'num': 1, 'prob': label["from_system"][1]}}
                merge_objects_num[eid] = {objects_num: {'num': 1, 'prob': label["objects_num"][1]}}
                merge_action[eid] = {action: {'num': 1, 'prob': label["action"][1]}}
            else:
                if from_system_num in merge_from_system[eid].keys():
                    merge_from_system[eid][from_system_num]['prob'] += label["from_system"][1]
                    merge_from_system[eid][from_system_num]['num'] += 1
                else:
                    merge_from_system[eid][from_system_num]['prob'] = label["from_system"][1]
                    merge_from_system[eid][from_system_num]['num'] = 1

                if objects_num in merge_objects_num[eid].keys():
                    merge_objects_num[eid][objects_num]['prob'] += label["objects_num"][1]
                    merge_objects_num[eid][objects_num]['num'] += 1
                else:
                    merge_objects_num[eid][objects_num]['prob'] = label["objects_num"][1]
                    merge_objects_num[eid][objects_num]['num'] = 1

                if action in merge_action[eid].keys():
                    merge_action[eid][action]['prob'] += label["action"][1]
                    merge_action[eid][action]['num'] += 1
                else:
                    merge_action[eid][action]['prob'] = label["action"][1]
                    merge_action[eid][action]['num'] = 1

    result = {}
    for eid, label_list in merge_dict.items():
        for key, value in label_list.items():
            if value == max(merge_dict[eid].values()):
                result[eid] = key
    #print(merge_from_system)
    from_system_result = {}
    for eid, label_list in merge_from_system.items():
        max_num = {'value': 0, 'num': 0, 'prob': 0}
        for key, value in label_list.items():
            if value['num'] == max_num['num'] and value['prob'] > max_num['prob']:
                max_num['value'] = key
                max_num['num'] = value['num']
                max_num['prob'] = value['prob']
            elif value['num'] > max_num['num']:
                max_num['value'] = key
                max_num['num'] = value['num']
                max_num['prob'] = value['prob']

        from_system_result[eid] = max_num['value']

    objects_num_result = {}
    for eid, label_list in merge_objects_num.items():
        max_num = {'value': 0, 'num': 0, 'prob': 0}
        for key, value in label_list.items():
            if value['num'] == max_num['num'] and value['prob'] > max_num['prob']:
                max_num['value'] = key
                max_num['num'] = value['num']
                max_num['prob'] = value['prob']
            elif value['num'] > max_num['num']:
                max_num['value'] = key
                max_num['num'] = value['num']
                max_num['prob'] = value['prob']
       
        objects_num_result[eid] = max_num['value']

    action_result = {}
    for eid, label_list in merge_action.items():
        max_num = {'value': 0, 'num': 0, 'prob': 0}
        for key, value in label_list.items():
                if value['num'] == max_num['num'] and value['prob'] > max_num['prob']:
                    max_num['value'] = key
                    max_num['num'] = value['num']
                    max_num['prob'] = value['prob']
                elif value['num'] > max_num['num']:
                    max_num['value'] = key
                    max_num['num'] = value['num']
                    max_num['prob'] = value['prob']

        action_result[eid] = max_num['value']

    return result, from_system_result, objects_num_result, action_result


def main(args):
    dialogs = json.load(open(args['split_path'], 'r'))
    output_path = open(args['save_path'], 'w+')

    dialogs_output = []

    output_result, from_system_result, objects_num_result, action_result = embedding_prediction(args['step1_pred_txt'])
    #print(output_result)
    for dialog_id, dialog_datum in enumerate(dialogs["dialogue_data"]):
        dialog_idx = dialog_datum["dialogue_idx"]
        predictions = []
        for turn_id, turn_datum in enumerate(dialog_datum["dialogue"]):
            if 'disambiguation_label' in turn_datum.keys():
                eid = str(dialog_id * 100 + turn_id)
                predictions.append({"turn_id": turn_datum['turn_idx'], "disambiguation_label": output_result[eid]})
        dialogs_output.append({"dialog_id": dialog_idx, "predictions": predictions})

    json.dump(dialogs_output, output_path)
    json.dump(from_system_result, open(os.path.join(args['save_path_for_other_step'], args['do_predict_split'] + '_from_system_embedding.json'), 'w+'))
    json.dump(objects_num_result, open(os.path.join(args['save_path_for_other_step'], args['do_predict_split'] + '_objects_num_embedding.json'), 'w+'))
    json.dump(action_result, open(os.path.join(args['save_path_for_other_step'], args['do_predict_split'] + '_action_embedding.json'), 'w+'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step1-pred-txt", help="Path dir to prediction of step1"
    )
    parser.add_argument(
        "--split-path", help="Process SIMMC file of test phase"
    )
    parser.add_argument(
        "--save-path",
        required=True,
        help="Path to save SIMMC dataset",
    )
    parser.add_argument(
        "--save-path-for-other-step",
        required=True,
        help="Path to save embedding data of other task",
    )
    parser.add_argument(
        "--do-predict-split",
        required=True,
        help="devtest or teststd",
    )

    try:
        parsed_args = vars(parser.parse_args())
        main(parsed_args)
    except (IOError) as msg:
        parser.error(str(msg))
