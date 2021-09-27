import json
import copy
import argparse
from util import utils


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
    
    
def step2_process(step2_target_data, step2_pred_data):
    result = {}
    for index, line in enumerate(step2_pred_data):
        line_json = json.loads(line.strip())
        eid = str(line_json['eid'])
        label_predict = int(line_json['labels'])
        label_prob = max([float(l) for l in line_json['label_prob']])
        if label_predict == 1:
            taget_list = step2_target_data[index].split('\t')
            slot_key = taget_list[3]
            slot_value = taget_list[4]
            
            if eid in result.keys():
                if slot_key in result[eid].keys():
                    result[eid][slot_key].append((slot_value, label_prob))
                else:
                    result[eid][slot_key] = [(slot_value, label_prob)]
        else:
            # result[eid] = {slot_key: [(slot_value, label_prob)]}
            continue
    for eid, value in result.items():
        for slot_key, sub_list in value.items():
            result[eid][slot_key] = sort(sub_list, key=lambda x: x[1], reversed=True)
    return result
    
    
def parse_prediction(args):
    slot_mapping, mapping_slot, action_mapping, mapping_action = get_label_mapping(args['slot_mapping_path'], args['action_mapping_path'])
    step1_pred_data = open(args['step1_pred_txt'], 'r').readlines()
    step2_pred_data = open(args['step2_pred_txt'], 'r').readlines()
    step2_target_data = open(args['step2_traget_txt'], 'r').readlines()

    step2_pred_dict = step2_process(step2_target_data, step2_pred_data)
    # print(step2_pred_dict)
    output_result = {}
    for index, line in enumerate(step1_pred_data):
        if line:
            pred_json = json.loads(line)
            pred_eid = str(pred_json["eid"])
            pred_action = pred_json["action"]
            pred_action_prob = pred_json["action_prob"]
            pred_disambiguate= pred_json["disambiguate"]
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
            try:
                for slot_key in step2_pred_dict[pred_eid].items():
                    if slot_key == 'request_slots':
                        for slot_value, slot_prob in step2_pred_dict[pred_eid][slot_key]:
                            if slot_prob > 0.5:
                                output_result[pred_eid]['act_attributes']['request_slots'].append(slot_value)
                    else:
                        output_result[pred_eid]['act_attributes']['slot_values'][slot_key] = step2_pred_dict[pred_eid][slot_key][0]
            except:
                continue
    return output_result

    
    
def main(args):
    dialogs = json.load(open(args['split_path'], 'r'))
    output = open(args['save_path'], 'w+')
    dialogs_output = copy.deepcopy(dialogs)

    output_result = parse_prediction(args)
    # print(output_result)
    for dialog_id, dialog_datum in enumerate(dialogs["dialogue_data"]):
        for turn_id, turn_datum in enumerate(dialog_datum["dialogue"]):
            # print(turn_datum["transcript"])
            eid = str(dialog_id*100 + turn_id)
            print(eid, output_result[eid])
            dialogs_output["dialogue_data"][dialog_id]["dialogue"][turn_id]["transcript_annotated"] = output_result[eid]
            dialogs_output["dialogue_data"][dialog_id]["dialogue"][turn_id]["system_transcript_annotated"] = {}
    json.dump(dialogs_output, output)
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step1-pred-txt",  help="Path to prediction of step1"
    )
    parser.add_argument(
        "--step2-pred-txt",  help="Path to prediction of step2"
    )
    parser.add_argument(
        "--step2-traget-txt",  help="Path to target of step2"
    )
    parser.add_argument(
        "--slot-mapping-path", help="File of slot mapping"
    )
    parser.add_argument(
        "--action-mapping-path", help="File of action mapping"
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