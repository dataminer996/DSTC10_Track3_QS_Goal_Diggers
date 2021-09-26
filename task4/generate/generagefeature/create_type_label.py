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
        for idname in ids:
            id_dic[idname] = len(os.listdir(os.path.join(dirname,typename,idname)))
        type_dic[typename] = len_ids
    print(type_dic)
    print(id_dic)
    return type_dic,id_dic

def create_label(dirname):
    typelist = os.listdir(dirname)
    type_dic = {}
    id_dic = {}
    filename_list = []
    label_list = []
    id_list = []
    for typename in typelist:
        # d = {}
        ids = os.listdir(os.path.join(dirname,typename))
        len_ids = len(ids)
        for idname in ids:
            filenames = os.listdir(os.path.join(dirname,typename,idname))
            for filename in filenames:
                filename_list.append(filename)
                label_list.append(typename)
                id_list.append(idname)

    return filename_list,label_list,id_list

dirname = sys.argv[1]
outputfile = sys.argv[2]
filename_list,label_list,id_list = create_label(dirname)
with  open(outputfile,'w') as tf:
   for i in range(len(filename_list)):
       label = label_list[i].replace(',','_')
       label = label.replace(' ','_')
       tf.writelines(filename_list[i]+","+str(label)+"\n")

get_type_id_img(sys.argv[1])
