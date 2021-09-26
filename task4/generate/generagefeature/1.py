import os
import json
import sys

def get_type_id_img(dirname):
    typelist = os.listdir(dirname)
    type_dic = {}
    id_dic = {}
    for typename in typelist:
        # d = {}
        ids = os.listdir(os.path.join(dirname,typename))
        len_ids = len(ids)
        for objid in ids:
            id_dic[objid] = len(os.listdir(os.path.join(dirname,typename,objid)))
        type_dic[typename] = len_ids
    print(type_dic)
    print(id_dic)
    return type_dic,id_dic
get_type_id_img(sys.argv[1])
