import codecs
def read_obj_fuction():
        
        global obj_function_fashion,obj_function_furniture
        dir_fashion = './data/fashion_result.txt'
        dir_furniture = './data/furniture_result.txt'
    
        txt = codecs.open(dir_fashion, mode='r', encoding='utf-8')
        did_all = []
        obj_id_all = []
        type_all = []

        for line in txt:
            #print(line)
            line = line.strip()
            imagetemp,temptypeobj = line.split("_objid_")
            scence = imagetemp[4:]
            objid,objtype = temptypeobj.split(".jpg,")
            #print(objtype)
            if objtype == "tank_top":
                print("find the tank_top")
                objtype = "tank top"
            if objtype == "shirt__vest":
                print("find the shirt_vest")
                objtype = "shirt vest"
            did_all.append(scence)
            type_all.append(objtype)
            obj_id_all.append(int(objid))


read_obj_fuction()
