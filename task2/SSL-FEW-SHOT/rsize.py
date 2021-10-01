import cv2
import os
import sys

def resize(load_dir, save_dir, size):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for jpg in os.listdir(load_dir):
        # i = png.split('.')[0]
        img = cv2.imread(os.path.join(load_dir, jpg))
        img = cv2.resize(img, (size, size))
        cv2.imwrite(os.path.join(save_dir, jpg), img)

resize(sys.argv[1],sys.argv[2],84)
