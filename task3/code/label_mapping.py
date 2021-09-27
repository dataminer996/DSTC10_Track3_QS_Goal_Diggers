import json
import pickle


mark_dict = {"label_list": ['0', '1'],
             "action_list": ['REQUEST:GET', 'REQUEST:COMPARE', 'INFORM:REFINE', 'INFORM:GET', 'ASK:GET',
                            'INFORM:DISAMBIGUATE', 'REQUEST:ADD_TO_CART'],
             "slot": {}}

SPLIT = ['train', 'dev', 'devtest']
count = 0
for split in SPLIT:
    with open('./data/simmc2_dials_dstc10_' + split + '.json', 'r', encoding='utf-8') as f:
        dialogs = json.load(f)
        for dialog_id, dialog_datum in enumerate(dialogs["dialogue_data"]):
            history = []
            for turn_datum in dialog_datum["dialogue"]:
                # slot
                for k, v in turn_datum["transcript_annotated"]['act_attributes']["slot_values"].items():
                    v = str(v)
                    if k not in mark_dict["slot"].keys():
                        try:
                            mark_dict["slot"][k] = [v.lower()]
                        except:
                            continue
                    else:
                        if v.lower() not in mark_dict["slot"][k]:
                            try:
                                mark_dict["slot"][k].append(v.lower())
                            except:
                                continue
                            
                request_slot = turn_datum["transcript_annotated"]['act_attributes']["request_slots"]
                if "request_slots" not in mark_dict["slot"].keys():
                    mark_dict["slot"]["request_slots"] = request_slot
                else:
                    for r in request_slot:
                        if r.lower() not in mark_dict["slot"]["request_slots"]:
                            mark_dict["slot"]["request_slots"].append(r.lower())

json.dump(mark_dict, open('./data/label_mapping.json', 'w+', encoding='utf-8'), ensure_ascii=False, indent=4)
print(mark_dict['slot'])

