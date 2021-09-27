import json

slot_candidate = {}
for file_name in ['../data/fashion_prefab_metadata_all.json', '../data/furniture_prefab_metadata_all.json']:
    data = json.load(open(file_name, 'r'))
    for key, value in data.items():
        for slot_key, slot_value in value.items():
            if len(str(slot_value))> 0:
                if slot_key in slot_candidate.keys():
                    if slot_value not in slot_candidate[slot_key]:
                        slot_candidate[slot_key].append(slot_value)
                else:
                    slot_candidate[slot_key] = [slot_value]

for file_name in ['../data/simmc2_dials_dstc10_train.json']:
    train_data = json.load(open(file_name, 'r'))
    for value in train_data['dialogue_data']:
        for e in value['dialogue']:
            for slot_key, slot_value in e['transcript_annotated']['act_attributes']['slot_values'].items():
                if len(str(slot_value))> 0:
                    if slot_key in slot_candidate.keys():
                        if slot_value not in slot_candidate[slot_key]:
                            slot_candidate[slot_key].append(slot_value)
                    else:
                        slot_candidate[slot_key] = [slot_value]
            for slot_value in e['transcript_annotated']['act_attributes']['request_slots']:
                if len(str(slot_value))> 0:
                    if 'request_slots' in slot_candidate.keys():
                        if slot_value not in slot_candidate['request_slots']:
                            slot_candidate['request_slots'].append(slot_value)
                    else:
                        slot_candidate['request_slots'] = [slot_value]

print(slot_candidate)
output = {}
for k, v in slot_candidate.items():
    if k in ['customerRating', 'price', 'customerReview']:
        tmp = []
        for e in v:
            try:
                e = float(e)
            except:
                tmp.append(e)
        output[k] = tmp
    else:
        output[k] = v

json.dump(output, open('slot_candidate.json', 'w+'), indent=4)