import os
import cv2
import sys

def png2jpeg(load_dir,save_dir):
    # load_dir = r'C:\simmc_last_train_dev\train'
    # save_dir = r'C:\simmc_last_train_dev\train'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for png in os.listdir(load_dir):
        # i = png.split('.')[0]i
      try:
        img = cv2.imread(os.path.join(load_dir, png))
        cv2.imwrite(os.path.join(save_dir, png.replace('png', 'jpg')), img)
      except:
        print(os.path.join(load_dir, png))     

png2jpeg(sys.argv[1],sys.argv[2])
