from util import utils
import random
import argparse
import copy
import json
import os

NEG_NUM = 3
#SPLITS = ["train", "dev", "devtest"]
#SPLITS = ["devtest"]
def get_label_mapping(args):
    slot_mapping = utils.load_pickle(args['slot_mapping'])
    mapping_slot = {}
    for key, value in slot_mapping.items():
        mapping_slot[value] = key

    action_mapping = utils.load_pickle(args['action_mapping'])
    mapping_action = {}
    for key, value in action_mapping.items():
        mapping_action[value] = key
    return slot_mapping, mapping_slot, action_mapping, mapping_action

def get_slot_candidate(args):
    slot_candidate = json.load(open(args['slot_candidate'], 'r'))
    return slot_candidate


def parse_slot_key(args, mapping_slot):
    read_path = args[f"step1_best_pred"]
    print(f"Reading: {read_path}")
    result = {}
    with open(read_path, "r") as file_id:
        lines = file_id.readlines()
        for line in lines:
            data_json = json.loads(line.strip())
            eid = data_json['eid']
            tmp = []
            for s in range(len(data_json['slot'])):
                if int(data_json['slot'][s]) == 1:
                    slot_key = mapping_slot[s]
                    tmp.append(slot_key)
            result[eid] = tmp
    return result

def train(args, split, slot_candidate):
        count = 0
        read_path = args[f"simmc_json"]
        print(f"Reading: {read_path}")
        with open(read_path, "r") as file_id:
            dialogs = json.load(file_id)
        target_path = args[f"target_txt"]
        target_json = {}
        with open(target_path, "r") as file_id:
            lines = file_id.readlines()
            for line in lines:
                if line:
                    chars, pos_tags, _, _, _, _, _, eid = line.strip().split('\t')
                    target_json[eid] = [chars, pos_tags]
        # Reformat into simple strings with positive and negative labels.
        # (dialog string, label)
        save_path = os.path.join(
            args["save_path"], f"{split}.txt"
        )
        print(f"Saving: {save_path}")
        with open(save_path, "w", encoding='utf-8') as file_id:
            for dialog_id, dialog_datum in enumerate(dialogs["dialogue_data"]):
                history = []
                for turn_id,turn_datum in enumerate(dialog_datum["dialogue"]):
                    # print(turn_datum["transcript"])
                    eid = dialog_id*100+turn_id
                    history.append(turn_datum["transcript"])
                    #nlp_result = standford_nlp(''.join(history))
                    nlp_result = target_json[str(eid)]
                    
                    # act label
                    action_label = turn_datum["transcript_annotated"]["act"]
                    
                    #response
                    response = turn_datum["system_transcript"]
                    count += 1

                    request_slot = turn_datum["transcript_annotated"]['act_attributes']["request_slots"]
                    # slot_key, value
                    for k, v in turn_datum["transcript_annotated"]['act_attributes']["slot_values"].items():
                        if k.lower() in map(lambda x:x.lower(), request_slot):
                            continue 
                        k = str(k)
                        v = str(v)
                        file_id.write('\t'.join([nlp_result[0], nlp_result[1],
                                            action_label,  k, v, '1', str(eid)]) + '\n')
                        # print('\t'.join(['#'.join(nlp_result['token']), '#'.join(nlp_result['pos']),
                        #                     action_label,  k, v, '1']))
                        
                        try:
                            for c_k, c_v in slot_candidate.items():
                                if c_k.lower() == k.lower():

                                    candidate = [str(n) for n in slot_candidate[c_k] if str(n).lower()!=v.lower()]
                            random.shuffle(candidate)
                            for e in candidate[:NEG_NUM]:
                                file_id.write('\t'.join([nlp_result[0], nlp_result[1],
                                                 action_label, k, e, '0', str(eid)]) + '\n')
                                # print('\t'.join(['#'.join(nlp_result['token']), '#'.join(nlp_result['pos']),
                                #                  action_label, k, e, '0']))
                        except Exception as e:
                            print(e)
                            continue

                    for each in request_slot:
                        file_id.write('\t'.join([nlp_result[0], nlp_result[1],
                                            action_label,  'request_slots', each, '1', str(eid)]) + '\n')
                        try:
                            candidate = [n for n in slot_candidate['request_slots'] if n.lower() not in map(lambda x:x.lower(), request_slot)]
                            random.shuffle(candidate)
                            for e in candidate[:NEG_NUM]:
                                file_id.write('\t'.join([nlp_result[0], nlp_result[1],
                                                 action_label, 'request_slots', e, '0', str(eid)]) + '\n')
                        except Exception as e:
                            print(e)
                            continue

                    if count % 100 == 0:
                      print(count)

                    history.append(response)

def devtest(args, split, slot_candidate):
        count = 0
        read_path = args["simmc_json"]
        print("Reading: {}".format(read_path))
        with open(read_path, "r") as file_id:
            dialogs = json.load(file_id)

        with open(args["step1_action"], "r") as file_id:
            step1_action = json.load(file_id)

        target_json = {}
        target_path = args[f"target_txt"]
        with open(target_path, "r") as file_id:
            lines = file_id.readlines()
            for line in lines:
                chars, pos_tags, _, _, _, _, _, eid = line.strip().split('\t')
                target_json[eid] = [chars, pos_tags]
        slot_mapping, mapping_slot, action_mapping, mapping_action = get_label_mapping(args)
        slot_pred = parse_slot_key(args, mapping_slot)
        # Reformat into simple strings with positive and negative labels.
        # (dialog string, label)
        save_path = os.path.join(
            args["save_path"], "{}.txt".format(split)
        )
        print(f"Saving: {save_path}")
        with open(save_path, "w", encoding='utf-8') as file_id:
            for dialog_id, dialog_datum in enumerate(dialogs["dialogue_data"]):
                history = []
                for turn_id, turn_datum in enumerate(dialog_datum["dialogue"]):
                    # print(turn_datum["transcript"])
                    eid = str(dialog_id*100 + turn_id)
                    history.append(turn_datum["transcript"])
                    nlp_result = target_json[eid]
                    
                    # act label
                    action_label = mapping_action[int(step1_action[eid])]
                   
                    #response
                    try:
                        response = turn_datum["system_transcript"]
                    except:
                        response = ''
 
                    count += 1
                 
                    # slot_key, value
                    for slot in slot_pred[eid]:
                        for v in slot_candidate[slot]:
                            #print(slot, v)
                            file_id.write('\t'.join([nlp_result[0], nlp_result[1],
                                                 action_label, slot, str(v), '0', str(eid)]) + '\n')

                    if count % 1000 == 0:
                        print(count)
                    
                    history.append(response)
            # print(f"# instances [{split}]: {len(result_data)}")


def main(args):
    slot_candidate = get_slot_candidate(args)
    if args['split'] == 'train':
        train(args, 'train', slot_candidate)
    elif args['split'] == 'dev':
        train(args, 'dev', slot_candidate)
    else:
        devtest(args, args['split'], slot_candidate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", help="train,dev,devtest,teststd"
    )
    parser.add_argument(
        "--simmc_json", default='./data/simmc2_dials_dstc10_train.json', help="Path to SIMMC file"
    )
    parser.add_argument(
        "--target-txt",  help="Path to train target of step1"
    )
    parser.add_argument(
        "--step1-action",  help="Path to action embedding of step1"
    )
    parser.add_argument(
        "--save-path",
        required=True,
        help="Path to save SIMMC JSONs",
    )
    parser.add_argument(
        "--slot-candidate",
        required=True,
        help="Path to save SIMMC JSONs",
    )   
    parser.add_argument(
        "--step1-best-pred",
        required=True,
        help="Path to save SIMMC JSONs",
    )
    parser.add_argument(
        "--slot-mapping",
        required=True,
        help="Path of slot mapping",
    )
    parser.add_argument(
        "--action-mapping",
        required=True,
        help="Path of action mapping",
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)


