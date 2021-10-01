import os
import sys
label_dic = {}

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


with open(sys.argv[1],'r') as tfr:
#   with open(sys.argv[2],'w') as tfw:
      lines = tfr.readlines()
      for line in lines:
        #  print(line)
          a,label =line.split(',')
          if label in label_dic.keys():
              label_dic[label] = label_dic[label] + 1            
          else:
              label_dic[label] = 1
print(label_dic) 


#if label_dic < 100:
root_dir = sys.argv[3]     
#with open(sys.argv[1],'r') as tfr:
with open(sys.argv[2],'w') as tfw:
      for label in label_dic.keys():
          if label.istitle():
              for line in lines:
#        #  print(line)
                 a,label_s = line.split(',')
                 if label == label_s:
                     rich_img(root_dir + "/" + a,root_dir)
                     for i in range(1,11):
                        line1 = line.replace('.jpg',"_"+str(i)+ ".jpg")   
                        tfw.writelines(line1)  
       #          if label in label_dic.keys():
#              if label_dic[label] < 40 or label[0].isupper():
                     
 #                 pass
                  #  copynum = int(100/label_dic[label]) + 1
                  #  for i in range(copynum):
                  #      tfw.writelines(line) 
 #             else:
 #                       tfw.writelines(line) 

                    
                           

 
