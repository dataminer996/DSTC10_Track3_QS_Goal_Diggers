import os
import sys
#import Image
from PIL import Image 


def rotate_img(img_path, num):
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = img.rotate(num, expand=True)
    return img

def tb_mirrotrans(img_path):
    img = Image.open(img_path)
    img = img.convert('RGB')
    return img.transpose(Image.FLIP_TOP_BOTTOM)

def lr_mirrotrans(img_path):
    img = Image.open(img_path)
    img = img.convert('RGB')
    return img.transpose(Image.FLIP_LEFT_RIGHT)
def rich_img(load_img, dump_dir):
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    count = 1
    for num in range(0, 360, 45):
        tmp = rotate_img(load_img, num)
        tmp.save(os.path.join(dump_dir, os.path.split(load_img.replace('.jpg', f'_{count}.jpg'))[1]))
        count += 1
    tmp = lr_mirrotrans(load_img)
    tmp.save(os.path.join(dump_dir, os.path.split(load_img.replace('.jpg', f'_{count}.jpg'))[1]))
    count += 1
    tmp = tb_mirrotrans(load_img)
    tmp.save(os.path.join(dump_dir, os.path.split(load_img.replace('.jpg', f'_{count}.jpg'))[1]))
rich_img(sys.argv[1],sys.argv[2])
