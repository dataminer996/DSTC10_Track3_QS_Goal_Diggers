import pickle
import json
import sys

def get_retrieval(retrieval_path, mode_path):
    with open(retrieval_path, 'r') as f:
        retrieval = json.load(f)
    with open(mode_path, 'r') as f:
        mode_json = json.load(f)
    idx_type = {}
    for itm in mode_json.get('dialogue_data'):
        idx_type[itm.get('dialogue_idx')] = itm.get('domain')
    # furniture = retrieval.get('system_transcript_pool').get('furniture')
    # fasion = retrieval.get('system_transcript_pool').get('fashion')
    retrieval_candidates = retrieval.get('retrieval_candidates')
    for num, itm in enumerate(retrieval_candidates):
        dialogue_idx = itm.get('dialogue_idx')
        type = idx_type.get(dialogue_idx)
        text_list = retrieval.get('system_transcript_pool').get(type)
        for sub_itm in itm.get('retrieval_candidates'):
            turn_idx = sub_itm.get('turn_idx')
            index_list = sub_itm.get('retrieval_candidates')
            gt_index = sub_itm.get('gt_index')
            for i, index in enumerate(index_list):
                text = text_list[index]
                label = 1 if i == gt_index else 0
                yield num, turn_idx, text, label,index


#path1 = './data/simmc2_dials_dstc10_devtest_retrieval_candidates.json'
#path2 = './data/simmc2_dials_dstc10_devtest.json'
#data = list(get_retrieval(path1, path2))
# print(data)
#bin_file = sys.argv[1]
#with open(bin_file, 'wb') as f:
#    pickle.dump(data, f)
