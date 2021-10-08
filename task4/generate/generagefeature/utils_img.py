#import Process
import pickle
import os
import cv2
import sys
import copy
import itertools
from PIL import Image
sys.path.append(os.getcwd())
import numpy as np



os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
def kj_feature(bbox, image):
    print('bbox con:',bbox)
    x1, y1, x2, y2 = bbox
    imagex, imagey = image
    newa = [x1 / imagex, y1 / imagey, x2 / imagex, y2 / imagey, (x2 - x1) / imagex, (y2 - y1) / imagey]
    return newa


def genimage_feature(img_path, bbox_list):
    os.system('rm -rf ./tools/mini_tsv/images/*')
    os.system('rm -rf ./imagebbox.npy')
    os.system('rm -rf ./imagesize.npy')
    os.system('rm -rf ./featureout.npy')
    os.system('rm -rf ./output/X152C5_test/inference/vinvl_vg_x152c4/predictions.*')
    os.system('rm -rf ./tools/mini_tsv/visualgenome/train.*')

    # subprocess.run(f'cp "{img_path}" "./tools/mini_tsv/"')
    os.system(f'cp {img_path} ./tools/mini_tsv/images')
    os.system('python ./tools/mini_tsv/tsv_demo.py')
    
    tmp = [0, 0]
    size = list(cv2.imread(img_path).shape[:2][::-1])
    tmp.extend(size)
    bbox_list.insert(0, tmp)
   # print("bbox_list",size,bbox_list)
    with open("log.txt",'a') as tf:
    #     line = "image_features shape" + str(new_image_feaures.shape)
     #    tf.writelines(line)
         line = "start\r\n"
         tf.writelines(line)
         line = "bbox_listtest" + str(size) + str(np.array(bbox_list).shape) + "\r\n"
         tf.writelines(line)
    #sys.exit(1)
    np.save("imagebbox.npy", bbox_list)
    # size = tuple(bbox_list[0][-2:])
    np.save("imagesize.npy", size)
    #subprocess.run('python tools/test_sg_net.py --config-file sgg_configs/vgattr/vinvl_x152c4.yaml TEST.IMS_PER_BATCH 2 MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth MODEL.ROI_HEADS.NMS_FILTER 1 MODEL.ROI_HEADS.SCORE_THRESH 0.2 DATA_DIR "./tools/mini_tsv/" TEST.IGNORE_BOX_REGRESSION True MODEL.ATTRIBUTE_ON True TEST.OUTPUT_FEATURE True')
    r = os.popen("./run.sh")
    text = r.read()
    r.close()

    #os.system('./run.sh')
    #Process.run('./run.sh')
#    cmdline = "python tools/test_sg_net.py --config-file sgg_configs/vgattr/vinvl_x152c4.yaml TEST.IMS_PER_BATCH 2 MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth MODEL.ROI_HEADS.NMS_FILTER 1 MODEL.ROI_HEADS.SCORE_THRESH 0.2 DATA_DIR ./tools/mini_tsv/ TEST.IGNORE_BOX_REGRESSION True MODEL.ATTRIBUTE_ON True TEST.OUTPUT_FEATURE True"
 #   print(cmdline)
 #   os.system(cmdline)
    #subprocess.run(['python', 'tools/test_sg_net.py', '--config-file', 'sgg_configs/vgattr/vinvl_x152c4.yaml', 'TEST.IMS_PER_BATCH', 2, 'MODEL.WEIGHT', 'pretrained_model/vinvl_vg_x152c4.pth', 'MODEL.ROI_HEADS.NMS_FILTER', 1, 'MODEL.ROI_HEADS.SCORE_THRESH', '0.2', 'DATA_DIR', "./tools/mini_tsv/", 'TEST.IGNORE_BOX_REGRESSION', 'True', 'MODEL.ATTRIBUTE_ON', 'True', 'TEST.OUTPUT_FEATURE', 'True'])


    image_features = np.load('./featureout.npy')
    #print("end bbox list",bbox_list)
    new_image_features = []
    for num, image_feature in enumerate(image_features):
        spatial_feature = kj_feature(bbox_list[num], size)
        image_feature = np.append(image_feature, spatial_feature)
        new_image_features.append(image_feature)
    new_image_features = np.array(new_image_features)
    _, img_name = os.path.split(img_path)
    with open("log.txt",'a') as tf:
         line = "end\r\n"
         tf.writelines(line)
         line = "image_features shape" + str(new_image_features.shape) + "\r\n"
         tf.writelines(line)
         line = "bbox_list shape" +  str(np.array(bbox_list).shape) + "\r\n"
         tf.writelines(line)
         line = "======end\r\n"
         tf.writelines(line)

    print("image_features shape",new_image_features.shape,np.array(bbox_list).shape)
    #picklefile = './img_feature.bin'
    #with open(picklefile, 'ab') as f:
    #    pickle.dump({'img_name': img_name, 'bbox_list': bbox_list, 'img_feature': new_image_features}, f)
    return new_image_features

image_featurea={}
image_features=[]
image_featurea['img_name'] = ""
image_featurea['bbox_list'] = ""
image_featurea['img_feature'] = ""

def find_imagekey(img_path, bbox_list):
  global image_features
  #print("img_path, bbox_list",img_path, bbox_list)
  for i in range(len(image_features)):
     #print("image_features",image_features[i]['img_name'],image_features[i]['bbox_list'])
     if image_features[i]['img_name'] == img_path and image_features[i]['bbox_list'] == bbox_list:
          return i
  return None

def readimagefrombin_feature_old(img_path, bbox_list):
    # img_dir = '/inputs'
    # img_path = os.path.join(img_dir, imagename)
    #picklefile = './img_feature.bin'
    global image_features
    global image_featurea
    image_featureb = {}
    #if os.path.exists(picklefile):
    index = find_imagekey(img_path, bbox_list)
    if index is not None:
        return image_features[index]['img_feature']
    else:
        print("don't find the index")
        #return
        bbox_list_new_1 = copy.deepcopy(bbox_list)
        bbox_list_new_2 = copy.deepcopy(bbox_list)
        image_featureb['img_name']  = img_path
        image_featureb['bbox_list'] = bbox_list_new_1
        feature = genimage_feature(img_path, bbox_list_new_2)
        image_featureb['img_feature'] = feature
        image_features.append(image_featureb)
        return feature

def readimagefrombin_feature(img_path, bbox_list, flags, paths):
    # img_dir = '/inputs'
    # img_path = os.path.join(img_dir, imagename)
    #picklefile = './img_feature.bin'
    global image_features
    global image_featurea
    global allgoodfeature

    image_featureb = {}
    #if os.path.exists(picklefile):
    index = find_imagekey(img_path, bbox_list)
    if index is not None:
        feature =  image_features[index]['img_feature']
    else:
        print("don't find the index")
        #return
        bbox_list_new_1 = copy.deepcopy(bbox_list)
        bbox_list_new_2 = copy.deepcopy(bbox_list)
        image_featureb['img_name']  = img_path
        image_featureb['bbox_list'] = bbox_list_new_1
        feature = genimage_feature(img_path, bbox_list_new_2)
        image_featureb['img_feature'] = feature
        image_features.append(image_featureb)
    feature = np.array(feature)
    for i in range(len(flags)):
        if flags[i] == False: 
           print("replaceobj start", paths[i])
           for goodpath,goodfeature in allgoodfeature:
               if goodpath == paths[i]:
                      feature[i][:2048] = goodfeature[:2048]
                      print("replaceobj sus", goodpath)
    return feature 
        


def readimage_feature(img_path, bbox_list):
    global image_features
    global image_feature
    # img_dir = '/inputs'
    # img_path = os.path.join(img_dir, imagename)
    #picklefile = './img_feature.bin'
    #if os.path.exists(picklefile):
    if image_feature['img_name'] == img_path and image_feature['bbox_list'] == bbox_list:
        print("find old one")
        return image_feature['img_feature']
    else:
        return
        image_feature['img_name']  = img_path
        image_feature['bbox_list'] = bbox_list

        #feature = genimage_feature(img_path, bbox_list)
        image_feature['img_feature'] = feature
        image_features.append(image_feature)
        return feature

def init_imagef(startid):
#    return
    global image_features
    filename = "image" + str(startid) + ".bin"
    if os.path.exists(filename):
           print("find the features file")
           with open(filename, 'rb') as f:
                 image_features = pickle.load(f)
def save_imagef(startid):
    return
    global image_features
    filename = "image" + str(startid) + ".bin"
    with open(filename, 'wb') as f:
        pickle.dump(image_features, f)
     
image_filename_list = []

def save_image( image_filename, bboxlist, typelist,objlist,save_dir_root):
    from PIL import Image
    global image_filename_list
    if image_filename in image_filename_list:
         return
   
    image_filename_list.append(image_filename)
   
    #TODO
#    save_dir_root = 'trainimage0910'
    assert len(bboxlist) == len(typelist) == len(objlist)
    with  Image.open(image_filename) as image:
      for bbox,type,obj in zip(bboxlist, typelist,objlist):
        obj = str(obj)
        pic = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        if not os.path.exists(os.path.join(save_dir_root,type,str(obj))):
            os.makedirs(os.path.join(save_dir_root,str(type),str(obj)))
        if not os.path.exists(os.path.join(save_dir_root,type,str(obj),type+'_'+str(obj)+'_'+ os.path.basename(image_filename))):
            #try:
               pic.save(os.path.join(save_dir_root,type,str(obj),type+'_'+str(obj)+'_'+ os.path.basename(image_filename)))
            #except:
               #print("error find the image",image_filename,bbox)
        else:
               print("find same the image",image_filename,bbox)
            

error_didobj_list = []


with open("path2name.bin",'rb') as f:
     path2namedict = pickle.load(f)
     #print(results)


def save_pred_image(dialogue_final,image_filename, bboxlist, typelist,objlist,save_dir_root):
    global image_filename_list
    global error_didobj_list
    if image_filename in image_filename_list:
        return

    image_filename_list.append(image_filename)

    # TODO
    #    save_dir_root = 'trainimage0910'
    assert len(bboxlist) == len(typelist) == len(objlist)
    with Image.open(image_filename) as image:
        for bbox, obj in zip(bboxlist,  objlist):
            obj = str(obj)
            if len(list(itertools.filterfalse(lambda x: x >= 0, bbox))) == 0 and bbox[2] > bbox[0] and bbox[3] > bbox[
                1]:#check bbox
                pic = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
#                if not os.path.exists(os.path.join(save_dir_root, type, str(obj))):
 #                   os.makedirs(os.path.join(save_dir_root, str(type), str(obj)))
                if not os.path.exists(os.path.join(save_dir_root,  'did_' + str(dialogue_final) +   '_objid_' + obj)):
                    # try:
                    pic = pic.convert('RGB')
                    save_path = os.path.join(save_dir_root,   'did_' + str(dialogue_final) +   '_objid_' + obj) + ".jpg"
                    print("save_path",save_path)
                    pic.save(save_path)
                    try:
                        with Image.open(save_path) as test:
                            print("find error image",dialogue_final, image_filename, bbox)
                            pass
                    except:
                        os.remove(save_path)
                        print("find error image",dialogue_final, image_filename, bbox)
                        error_didobj_list.append((dialogue_final,obj))    
                        
                # except:
                # print("error find the image",image_filename,bbox)
                else:
                    print("find same image",dialogue_final, image_filename, bbox)
            else:
                print("find error image",dialogue_final, image_filename, bbox)
                error_didobj_list.append((dialogue_final,obj))    
                #TODO


error_didobj_list = []
def save_pred_image_otherview(dialogue_final,image_filename, bboxlist, typelist,objlist,save_dir_root,prefab_final):
    global image_filename_list
    global error_didobj_list
    global path2namedict
    if image_filename in image_filename_list:
        return

    image_filename_list.append(dialogue_final)

    # TODO
    #    save_dir_root = 'trainimage0910'
    assert len(bboxlist) == len(typelist) == len(objlist)
    with Image.open(image_filename) as image:
        for bbox, obj, prefab in zip(bboxlist,  objlist,prefab_final):
            obj = str(obj)
            if prefab in path2namedict.keys():
                filename = path2namedict[prefab]
                os.system("/bin/cp   otherviewimage/" +  filename +  "  " + os.path.join(save_dir_root,  'did_' + str(dialogue_final) +   '_objid_' + obj + ".jpg"))
                print("replace image ",dialogue_final,obj)
                continue
            if len(list(itertools.filterfalse(lambda x: x >= 0, bbox))) == 0 and bbox[2] > bbox[0] and bbox[3] > bbox[
                1]:#check bbox
                pic = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
#                if not os.path.exists(os.path.join(save_dir_root, type, str(obj))):
 #                   os.makedirs(os.path.join(save_dir_root, str(type), str(obj)))
                if not os.path.exists(os.path.join(save_dir_root,  'did_' + str(dialogue_final) +   '_objid_' + obj + ".jpg")):
                    # try:
                    pic = pic.convert('RGB')
                    save_path = os.path.join(save_dir_root,   'did_' + str(dialogue_final) +   '_objid_' + obj) + ".jpg"
                    print("save_path",save_path)
                    pic.save(save_path)
                    try:
                        with Image.open(save_path) as test:
                            print("find error image",dialogue_final, image_filename, bbox)
                            pass
                    except:
                        os.remove(save_path)
                        print("find error image",dialogue_final, image_filename, bbox)
                        error_didobj_list.append((dialogue_final,obj))    
                        
                # except:
                # print("error find the image",image_filename,bbox)
                else:
                    print("find same image",dialogue_final, image_filename, bbox)
            else:
                print("find error image",dialogue_final, image_filename, bbox)
                error_didobj_list.append((dialogue_final,obj))    
                #TODO            
                
def save_error_bin(startid):
    with open("errorimagesave" + str(startid) + ".bin", 'wb') as f:
        pickle.dump(error_didobj_list, f)
      

if __name__ == '__main__':
	# global image_features
#    with open("test.bin", 'rb') as f:
#       image_features = pickle.load(f)
	#print("len of image_features",len(image_features))
     #   print("image_features",image_features[0]['img_name'],image_features[0]['bbox_list'])
    readimagefrombin_feature('./cloth_store_1416238_woman_10_0.png', [[23,33,100,100],[55,34,300,300]])
    readimagefrombin_feature('./cloth_store_1416238_woman_10_0.png', [[24,45,100,100],[55,34,300,300]])
    print(image_features) 
    with open("test.bin", 'wb') as f:
        pickle.dump(image_features, f)
    
