import json
from util import utils
import argparse
import tensorflow as tf


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


def get_slot_candidate():
    return json.load(open('./data/slot_candidate.json', 'r'))
    

def main(args):
    slot_mapping, mapping_slot, action_mapping, mapping_action = get_label_mapping(args['slot_mapping_path'], args['action_mapping_path'])
    target_data = tf.io.gfile.GFile(args['target_txt'], mode='r').readlines()
    pred_data = tf.io.gfile.GFile(args['pred_txt'], mode='r').readlines()
    output = tf.io.gfile.GFile(args['save_path'], mode='w+')
    slot_candidate = get_slot_candidate()

    count = 0
    for index, line in enumerate(target_data):
        line = line.strip().split('\t')
        if line and len(line) == 8:
            chars, pos_tags, _, _, _, _, _, eid = line
            pred_json = json.loads(pred_data[index])
            pred_eid = pred_json["eid"]
            pred_action = pred_json["action"]
            pred_action_prob = pred_json["action_prob"]
            pred_disambiguate= pred_json["disambiguate"]
            pred_disambiguate_prob = pred_json["disambiguate_prob"]
            pred_slot = pred_json["slot"]
            pred_slot_prob = pred_json["slot_prob"]

            pred_action_name = mapping_action[int(pred_action)]

            assert eid == pred_eid
            #print(eid)

            for i, each in enumerate(pred_slot_prob):
                each = float(each)
                # print(each)
                if each > 0.7:
                    slot_key = mapping_slot[i]
                    for slot_value in slot_candidate[slot_key]:
                        output.write('\t'.join([chars, pos_tags, pred_action_name, slot_key, slot_value, '0', eid])
                                     + '\n')
                        count += 1
                        if count % 1000 == 0:
                            print(count)
    output.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pred-txt",  help="Path to prediction of step1"
    )
    parser.add_argument(
        "--target-txt",  help="Path to target of step1"
    )
    parser.add_argument(
        "--slot-mapping-path", help="File of slot mapping"
    )
    parser.add_argument(
        "--action-mapping-path", help="File of action mapping"
    )
    parser.add_argument(
        "--split", default='devtest', help="Process SIMMC file of test phase"
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
