import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pickle
import torch

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
#IMAGE_PATH = osp.join(ROOT_PATH, 'data/miniimagenet/images')

IMAGE_PATH = osp.join(ROOT_PATH, 'data/miniimagenet/image0910jpg')
SPLIT_PATH = osp.join(ROOT_PATH, 'data/miniimagenet/split_0910')
#SPLIT_PATH = osp.join(ROOT_PATH, 'data/miniimagenet/split_0910')

class MiniImageNet(Dataset):
    """ Usage: 
    """
    def __init__(self, setname, args):
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        self.num_class = len(set(label))
        print("num_class",self.num_class)
        print("len data",len(self.data))

        # Transformation
        if args.model_type == 'ConvNet':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif args.model_type == 'ResNet':
            image_size = 80
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), 
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif args.model_type == 'AmdimNet':
            INTERP = 3
            post_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            self.transform = transforms.Compose([
                transforms.Resize(146, interpolation=INTERP),
                transforms.CenterCrop(128),
                post_transform
            ])
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
       # print("===========one item")
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        #print(image.shape)
        #print(image.shape)
        #print("+++++one item")

        return image, label


class MiniImageNet_Test(Dataset):
    """ Usage: 
    """
    def __init__(self, setname, args):
        
        SPLIT_PATH = args.split_path
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(wnid)
        #self.query = data


        self.data = data
        self.label = label
        self.num_class = len(set(label))
        print("test num_class",self.num_class)
        print("test len data",len(self.data))        

        csv_path = osp.join(SPLIT_PATH, "train" + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()]

        traindata = []
        trainlabel = []
        trainlb = -1

        self.trainwnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.trainwnids:
                self.trainwnids.append(wnid)
                trainlb += 1
            traindata.append(path)
            trainlabel.append(wnid)
        #self.query = data

        csv_path = osp.join(SPLIT_PATH, "train_all" + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()]

        traindata_all = []
        trainlabel_all = []
        trainlb_all = -1

        self.trainwnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.trainwnids:
                self.trainwnids.append(wnid)
                trainlb_all += 1
            traindata_all.append(path)
            trainlabel_all.append(wnid)



        self.shot_num = args.shot
        self.way_num = args.way
        self.qeury_num = args.query
 
        self.traindata = traindata
        self.trainlabel = trainlabel
        
        trainlabel_set = list(set(trainlabel))
        
        self.train_num_class = len(trainlabel_set)
        self.trainlabel_set = trainlabel_set

        print("train_num_class",self.train_num_class)
        print("len traindata",len(self.traindata))
        print("trainlabel_set",trainlabel_set)
        bin_path = osp.join(SPLIT_PATH, "trainlabelset" + '.bin')
        
        slot_data = []
        with open(bin_path,'wb') as fp:
                 pickle.dump(trainlabel_set,fp)

        
        print("args.model_type",args.model_type)
             
        # Transformation
        if args.model_type == 'ConvNet':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif args.model_type == 'ResNet':
            image_size = 80
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), 
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif args.model_type == 'AmdimNet':
            print("args.model_type AmdimNet",args.model_type)
            INTERP = 3
            post_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            self.transform = transforms.Compose([
                transforms.Resize(146, interpolation=INTERP),
                transforms.CenterCrop(128),
                post_transform
            ])

        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')

        slot_data_new = [] 
        slot_label = []
        
        slot_label_new = []
        for j in range(self.train_num_class):     
           num = 0
           for i in range(len(self.trainlabel)):             
             #for label_find in set(trainlabel):
             if trainlabel_set[j] ==  self.trainlabel[i]: 
                if num < self.shot_num:      
                   imagename = self.traindata[i]       
                   slot_label.append(self.trainlabel[i])
                   image = self.transform(Image.open(imagename).convert('RGB'))
                   #print(image)
                   slot_data.append(image.numpy())                   
                   num = num + 1
                else:
                   break
           if num < self.shot_num:      
            for i in range(len(trainlabel_all)):     
             if trainlabel_set[j] ==  trainlabel_all[i]: 
               if num < self.shot_num:      
                  imagename = traindata_all[i]       
                  slot_label.append(trainlabel_all[i])
                  image = self.transform(Image.open(imagename).convert('RGB'))
              #print(image)
                  slot_data.append(image.numpy())                   
                  num = num + 1
               else:
                  break

        print(slot_label)
        shot_num =  self.shot_num
        train_num_class = self.train_num_class
        for j in range(shot_num):
             for i in range(train_num_class):
                slot_data_new.append(slot_data[i*shot_num+j])
                slot_label_new.append(slot_label[i*shot_num+j])
        slot_data_new = np.array(slot_data_new)              
        slot_data = np.array(slot_data)         
        print(slot_data.shape)
        print(slot_data_new.shape)


        print(slot_label)
        print(slot_label_new)  

        print("len slot_data",slot_data.shape)          
        if len(slot_data) !=   self.train_num_class *  self.shot_num:
             print("++++++++++++++++++++++++find the error abput slot_data:",len(slot_data),",shot_num:",self.shot_num)
        self.slot_data_new   =  torch.from_numpy(slot_data_new)
        self.slot_data   =  torch.from_numpy(slot_data)
        print("len slot_data",self.slot_data.shape)                   
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
       # print("===========one item")
        path, label = self.data[i], self.label[i]
        query_image = self.transform(Image.open(path).convert('RGB'))
        query_image = query_image.reshape(1,3,128,128)
        #query_image = query_image.repeat(self.way_num,1,1).reshape(self.way_num,3,128,128)
        #print(query_image.shape)
        #print(query_image)
        #print("+++++one item")
        slot_image = self.slot_data_new
        #print(slot_image.shape)

        label_index =  self.trainlabel_set.index(label) 
        
        label_index =  torch.tensor(label_index, dtype=torch.int64)
        #label_index =  torch.tensor(label_index, dtype=torch.int64).repeat(self.way_num).reshape(self.way_num,)
        return i,(query_image,slot_image,label_index)      

