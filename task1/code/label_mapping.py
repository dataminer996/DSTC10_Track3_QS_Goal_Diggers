import json
import pickle


mark_dict = {"disambiguate_list": ['0', '1'],
             "action_list": ['REQUEST:GET', 'REQUEST:COMPARE', 'INFORM:REFINE', 'INFORM:GET', 'ASK:GET',
                            'INFORM:DISAMBIGUATE', 'REQUEST:ADD_TO_CART'],
             "slot_types": [],
             "pos_tag": []}

DOMAIN = ['fashion', 'furniture']
for domain in DOMAIN:
  with open('./data/' + domain + '_prefab_metadata_all.json', 'r', encoding='utf-8') as f:
      data = json.load(f)
      for _, value in data.items():
          for slot_key, v in value.items():
              if slot_key not in mark_dict["slot_types"]:
                  mark_dict["slot_types"].append(slot_key)
mark_dict["slot_types"].append("request_slots")

SPLIT = ['train', 'dev', 'devtest']
count = 0
for split in SPLIT:
    with open('./data/finetuning_data/chunk/' + split + '.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            pos_list = line.split('\t')[1].split('#')
            for pos in pos_list:
                if pos not in mark_dict['pos_tag']:
                    mark_dict['pos_tag'].append(pos)
                    count += 1

json.dump(mark_dict, open('./data/label_mapping.json', 'w+', encoding='utf-8'), ensure_ascii=False, indent=4)
print(len(mark_dict['slot_types']))

label_mapping = {label: i for i, label in enumerate(mark_dict["action_list"])}
print(label_mapping)
#utils.write_pickle(label_mapping, self._label_mapping_path)
with open('./data/models/electra_large/finetuning_tfrecords/chunk_tfrecords/chunk_action.pkl', "wb") as f:
    pickle.dump(label_mapping, f, -1)

label_mapping = {label: i for i, label in enumerate(mark_dict["slot_types"])}
print(label_mapping)
#utils.write_pickle(label_mapping, self._label_mapping_path)
with open('./data/models/electra_large/finetuning_tfrecords/chunk_tfrecords/chunk_slot_types.pkl', "wb") as f:
    pickle.dump(label_mapping, f, -1)

label_mapping = {label: i for i, label in enumerate(mark_dict["pos_tag"])}
with open('./data/models/electra_large/finetuning_tfrecords/chunk_tfrecords/chunk_pos_tags.pkl', "wb") as f:
    pickle.dump(label_mapping, f, -1)