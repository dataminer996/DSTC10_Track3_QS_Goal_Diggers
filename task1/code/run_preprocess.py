import stanza
import random
import argparse
import copy
import json
import os
import re
import tensorflow as tf


en_nlp = stanza.Pipeline('en', processors='tokenize,pos')


def get_slot_candidate(slot_candidate_path):
    file_id = tf.io.gfile.GFile(slot_candidate_path, mode='r')
    return json.load(file_id)
    
    
def standford_nlp(sentence):
    global en_nlp
    doc = en_nlp(sentence)
    result = {'token': [],
              'pos': []}
    # print(doc)
    for sent in doc.sentences:
        # print('-----------------')
        # print(type(sent))
        result['token'] += [token.text for token in sent.tokens]
        result['pos'] += [word.xpos for word in sent.words]
    # print(result['pos'])
    return result


def main(args):
    NEG_NUM = 3
    count = 0
    split = args[f"split"]
    read_path = args[f"simmc_json"]
    print(f"Reading: {read_path}")
    
    with tf.io.gfile.GFile(read_path, mode='r') as simmc_file:
        dialogs = json.load(simmc_file)

    # Reformat into simple strings with positive and negative labels.
    # (dialog string, label)
    save_path = os.path.join(
        args["action_save_path"], f"{split}.txt"
    )
    print(f"Saving: {save_path}")
    file_id = tf.io.gfile.GFile(save_path, mode='w')
    
    if split in ['train', 'dev']:
        slot_save_path = os.path.join(
        args["slot_save_path"], f"{split}.txt"
    )
        print(f"Saving: {slot_save_path}")
        slot_file_id = tf.io.gfile.GFile(slot_save_path, mode='w')
        slot_candidate = get_slot_candidate(args["slot_candidate_path"])

    for dialog_id, dialog_datum in enumerate(dialogs["dialogue_data"]):
        history = []
        # dialogue_idx = dialog_datum["dialogue_idx"]
        for turn_id, turn_datum in enumerate(dialog_datum["dialogue"]):
            eid = dialog_id*100 + turn_id
            # print(turn_datum["transcript"])
            history.append(turn_datum["transcript"])
            text = re.compile('[\\x00-\\x08\\x0b-\\x0c\\x0e-\\x1f]').sub(' ', ''.join(history))
            nlp_result = standford_nlp(text)

            # disambiguate label
            try:
                disambiguate_label = turn_datum["disambiguation_label"]
            except:
                disambiguate_label = 2

            # act label
            action_label = turn_datum["transcript_annotated"]["act"]
            # request slot
            request_slot = turn_datum["transcript_annotated"]['act_attributes']["request_slots"]
            slot_values = turn_datum["transcript_annotated"]['act_attributes']["slot_values"]
            
            # from system
            from_system = turn_datum["transcript_annotated"]['act_attributes']['from_system']
            if from_system == '-1':
                from_system = '2'

            # slot_key/value
            # write step2 sample
            if split in ['train', 'dev']:
                slot_key = []
                for k, v in slot_values.items():
                    if k not in request_slot:
                        slot_key.append(k)
                        v = str(v)
                        slot_file_id.write('\t'.join(['#'.join(nlp_result['token']), '#'.join(nlp_result['pos']),
                                                action_label,  k, v, '1', str(eid)]) + '\n')
                            # print('\t'.join(['#'.join(nlp_result['token']), '#'.join(nlp_result['pos']),
                            #                     action_label,  k, v, '1']))
                            
                        try:
                            for c in slot_candidate.keys():
                                if c.lower() == k.lower():
                                    break
                            # print(k, v)
                            candidate = [str(n) for n in slot_candidate[c] if str(n).lower()!=str(v).lower()]
                            random.shuffle(candidate)
                            for e in candidate[:NEG_NUM]:
                                e = str(e)
                                slot_file_id.write('\t'.join(['#'.join(nlp_result['token']), '#'.join(nlp_result['pos']),
                                                 action_label, k, e, '0', str(eid)]) + '\n')
                                # print('\t'.join(['#'.join(nlp_result['token']), '#'.join(nlp_result['pos']),
                                #                  action_label, k, e, '0']))
                        except Exception as e:
                            print('1', e)
                            continue
                    
                if len(request_slot) > 0:
                    slot_key.append('request_slots')
                    for each in request_slot:
                        slot_file_id.write('\t'.join(['#'.join(nlp_result['token']), '#'.join(nlp_result['pos']),
                                                action_label,  'request_slot', each.lower(), '1', str(eid)]) + '\n')
                    try:
                        candidate = [n for n in slot_candidate.keys() if n.lower() not in [r.lower() for r in request_slot]]
                        random.shuffle(candidate)
                        for e in candidate[:NEG_NUM]:
                            slot_file_id.write('\t'.join(['#'.join(nlp_result['token']), '#'.join(nlp_result['pos']),
                                             action_label, 'request_slot', e, '0', str(eid)]) + '\n')
                    except Exception as e:
                        print('2', e)
                        continue
            else:
                slot_key = []
                for k, v in slot_values.items():
                    if k not in request_slot:
                        slot_key.append(k)

                if len(request_slot) > 0:
                    slot_key.append('request_slots')

            # objects
            objects = turn_datum["transcript_annotated"]['act_attributes']["objects"]
            objects_num = str(len(objects))
            
            # write step1 sample
            if split in ['train', 'dev']:
                file_id.write('\t'.join(['#'.join(nlp_result['token']), '#'.join(nlp_result['pos']),
                                     str(disambiguate_label), action_label,
                                     '#'.join(slot_key), from_system, objects_num, str(eid)]) + '\n')
            else:
                file_id.write('\t'.join(['#'.join(nlp_result['token']), '#'.join(nlp_result['pos']), '-1', '', '', '-1', '-1', str(eid)]) + '\n')
                
            if count % 1000 == 0:
                print(count)
            count += 1
            
            #response
            response = turn_datum["system_transcript"]
            history.append(response)
        # print(f"# instances [{split}]: {len(result_data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simmc_json", default='./data/simmc2_dials_dstc10_train.json', help="Path to SIMMC file"
    )
    parser.add_argument("--simmc_dev_json", default='./data/simmc2_dials_dstc10_dev.json',
                        help="Path to SIMMC dev file")
    parser.add_argument(
        "--simmc_devtest_json", default='./data/simmc2_dials_dstc10_devtest.json', help="Path to SIMMC devtest file"
    )
    parser.add_argument(
        "--split", default='train', help="Process SIMMC file"
    )
    parser.add_argument(
        "--action-save-path",
        required=True,
        help="Path to save SIMMC action dataset",
    )
    parser.add_argument(
        "--slot-save-path",
        required=True,
        help="Path to save SIMMC slot dataset",
    )
    parser.add_argument(
        "--slot-candidate-path",
        required=True,
        help="Path to slot candidate dataset",
    )
    
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
    # result = standford_nlp('Process finished with exit code')


