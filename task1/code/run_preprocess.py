import stanza
import random
import argparse
import copy
import json
import os
import re
import tensorflow as tf


en_nlp = stanza.Pipeline('en', processors='tokenize,pos')

    
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
    

    for dialog_id, dialog_datum in enumerate(dialogs["dialogue_data"]):
        history = []
        # dialogue_idx = dialog_datum["dialogue_idx"]
        for turn_id, turn_datum in enumerate(dialog_datum["dialogue"]):
            eid = dialog_id*100 + turn_id
            # print(turn_datum["transcript"])
            history.append(turn_datum["transcript"])
            text = re.compile('[\\x00-\\x08\\x0b-\\x0c\\x0e-\\x1f]').sub(' ', ''.join(history))
            nlp_result = standford_nlp(text)

            if split in ['train', 'dev']:
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
                file_id.write('\t'.join(['#'.join(nlp_result['token']), '#'.join(nlp_result['pos']),
                                     str(disambiguate_label), action_label,
                                     '#'.join(slot_key), from_system, objects_num, str(eid)]) + '\n')
            else:
                file_id.write('\t'.join(['#'.join(nlp_result['token']), '#'.join(nlp_result['pos']), '-1', '', '', '-1', '-1', str(eid)]) + '\n')
                
            if count % 1000 == 0:
                print(count)
            count += 1
            
            #response
            try:
                response = turn_datum["system_transcript"]
            except:
                response = ''
            history.append(response)
        # print(f"# instances [{split}]: {len(result_data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simmc_json", default='./data/simmc2_dials_dstc10_train.json', help="Path to SIMMC file"
    )
    parser.add_argument(
        "--split", default='train', help="Process SIMMC file"
    )
    parser.add_argument(
        "--action-save-path",
        required=True,
        help="Path to save SIMMC action dataset",
    )
    
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
    # result = standford_nlp('Process finished with exit code')


