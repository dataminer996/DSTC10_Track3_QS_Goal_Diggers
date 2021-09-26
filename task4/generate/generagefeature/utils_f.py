#import Process
import pickle
import os
import cv2
import sys
import copy
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
    
