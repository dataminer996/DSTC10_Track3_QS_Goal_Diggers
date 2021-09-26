# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 
from __future__ import absolute_import, division, print_function
import argparse
import os
import sys
sys.path.append(os.getcwd())

import base64
import os.path as op
import random, json
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import pickle

from oscar.utils.tsv_file import TSVFile
from oscar.utils.logger import setup_logger
from oscar.utils.misc import mkdir, set_seed
from transformersm.pytorch_transformers import BertTokenizer, BertConfig 
from transformersm.pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
from oscar.modeling.modeling_bert import ImageBertForSequenceClassification


class RetrievalDataset(Dataset):
    """ Image/Text Retrieval Dataset"""
    def __init__(self, tokenizer, args, split='train', is_train=True):
        """
        tokenizer: tokenizer to process caption text.
        args: configureation parameters including max_seq_length, etc.
        split: used to infer the data used for training or testing. 
             All files are in .pt format of a dictionary with image keys and 
             image features (pytorch tensors), captions (list of str, support multiple
             captions per image), labels (list of dictionary or str of all labels),

        """
        super(RetrievalDataset, self).__init__()
        if 0:
                     self.img_file = args.img_feat_file
                     caption_file = op.join(args.data_dir, '{}_captions.pt'.format(split))
                     self.img_tsv = TSVFile(self.img_file)
                     self.captions = torch.load(caption_file)
                     self.img_keys = list(self.captions.keys())  # img_id as int
                     if not type(self.captions[self.img_keys[0]]) == list:
                         self.captions = {k: json.loads(self.captions[k]) for k in self.img_keys}
             
                     # get the image image_id to index map
                     imgid2idx_file = op.join(op.dirname(self.img_file), 'imageid2idx.json')
                     self.image_id2idx = json.load(open(imgid2idx_file))  # img_id as string
                     
                     if args.add_od_labels:
                         label_data_dir = op.dirname(self.img_file)
                         label_file = os.path.join(label_data_dir, "predictions.tsv")
                         self.label_tsv = TSVFile(label_file)
                         self.labels = {}
                         for line_no in range(self.label_tsv.num_rows()):
                             row = self.label_tsv.seek(line_no)
                             image_id = row[0]
                             if int(image_id) in self.img_keys:
                                 results = json.loads(row[1])
                                 objects = results['objects'] if type(
                                     results) == dict else results
                                 self.labels[int(image_id)] = {
                                     "image_h": results["image_h"] if type(
                                         results) == dict else 600,
                                     "image_w": results["image_w"] if type(
                                         results) == dict else 800,
                                     "class": [cur_d['class'] for cur_d in objects],
                                     "boxes": np.array([cur_d['rect'] for cur_d in objects],
                                                       dtype=np.float32)
                                 }
                         self.label_tsv._fp.close()
                         self.label_tsv._fp = None
                         


                         if is_train:
                             self.num_captions_per_img = args.num_captions_per_img_train
                         else:
                             self.num_captions_per_img = args.num_captions_per_img_val
                             if args.eval_img_keys_file:
                                 # select a subset of image keys for evaluation. eg. COCO 1k and 5k
                                 # eval_img_keys_file is a list of image keys saved in tsv file
                                 with open(op.join(args.data_dir, args.eval_img_keys_file), 'r') as f:
                                     img_keys = f.readlines()
                                 self.img_keys = [int(k.strip()) for k in img_keys]
                                 self.captions = {k: self.captions[k] for k in self.img_keys}
                                 if args.add_od_labels:
                                     self.labels = {k: self.labels[k] for k in self.img_keys}
                 
                             if args.eval_caption_index_file:
                                 # hard negative image/caption indexs for retrieval re-rank setting.
                                 # useful for mini val set to monitor the performance during training.
                                 # However, it cannot be used together with cross image evaluation.
                                 self.has_caption_indexs = True
                                 assert not args.cross_image_eval 
                                 caption_index_file = op.join(args.data_dir, args.eval_caption_index_file)
                                 self.caption_indexs = torch.load(caption_index_file)
                                 if not type(self.caption_indexs[self.img_keys[0]]) == list:
                                     self.caption_indexs = {k: json.loads(self.caption_indexs[k]) for k in self.img_keys}
                             else:
                                 self.has_caption_indexs = False

        trainfile = args.data_dir + "/train.bin"
        devfile = args.data_dir + "/dev.bin"  
        print("trainfile",trainfile)
        print("devfile",devfile)

        with open(trainfile,'rb') as f:
             self.feature = pickle.load(f)
             random.shuffle(self.feature)
        print("train example num",len(self.feature))

        with open(devfile,'rb') as f:
             self.devfeature =  pickle.load(f)
        print("dev example num",len(self.devfeature))

        self.is_train = is_train
        self.output_mode = args.output_mode
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_length
        self.max_img_seq_len = args.max_img_seq_length
        self.args = args

    def get_image_caption_index(self, index):
        # return img_idx to access features and [img_key, cap_idx] to access caption
        if not self.is_train and self.args.cross_image_eval:
            img_idx = index // (self.num_captions_per_img * len(self.img_keys))
            cap_idx = index % (self.num_captions_per_img * len(self.img_keys))
            img_idx1 = cap_idx // self.num_captions_per_img
            cap_idx1 = cap_idx % self.num_captions_per_img
            return img_idx, [self.img_keys[img_idx1], cap_idx1]
        if not self.is_train and self.has_caption_indexs:
            img_idx = index // self.num_captions_per_img
            cap_idx = index % self.num_captions_per_img
            img_key1, cap_idx1 = self.caption_indexs[self.img_keys[img_idx]][cap_idx]
            return img_idx, [img_key1, cap_idx1]
        img_idx = index // self.num_captions_per_img
        cap_idx = index % self.num_captions_per_img
        return img_idx, [self.img_keys[img_idx], cap_idx]

    def get_label(self, index):
        img_idx, cap_idx = self.get_image_caption_index(index)
        return 1 if self.img_keys[img_idx] == cap_idx[0] else 0

    def get_od_labels(self, img_key):
        if self.args.add_od_labels:
            if type(self.labels[img_key]) == str:
                od_labels = self.labels[img_key]
            else:
                od_labels = ' '.join(self.labels[img_key]['class'])
            return od_labels

    def tensorize_example_old(self, text_a, img_feat,label_final, text_b=None, 
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):
        tokens_a = self.tokenizer.tokenize(text_a)
        tokens_b = self.tokenizer.tokenize(text_b)
        if len(tokens_a) > self.args.max_seq_length - 3 - len(tokens_b) :
            tokens_a = tokens_a[:(self.args.max_seq_length - 3 - len(tokens_b))]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) + 1)
        seq_a_len = len(tokens)
        if text_b:
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)
        seq_padding_len = self.max_seq_len - seq_len
        tokens += [self.tokenizer.pad_token] * seq_padding_len
        segment_ids += [pad_token_segment_id] * seq_padding_len
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # image features
        img_len = img_feat.shape[0]
        img_feat= torch.tensor(img_feat, dtype=torch.float32)
        label_final = torch.tensor(label_final, dtype=torch.int64)
        label_final_len = label_final.shape[0]
        if label_final_len > self.max_img_seq_len-1:
            label_final = torch.tensor(label_final[0 : self.max_img_seq_len-1, :])
            label_len = label_final.shape[0]   
            label_padding_len = 0 
        else:
            label_padding_len = self.max_img_seq_len - 1 - label_final_len
            padding_matrix = torch.zeros((label_padding_len))
            label_final = torch.cat((label_final, padding_matrix), 0)
            
        #padding_matrix = torch.zeros((self.args.max_seq_length+1))    
        #label_feat =     torch.cat((padding_matrix,label_feat), 0) 
        
        if img_len > self.max_img_seq_len:
            img_feat = torch.tensor(img_feat[0 : self.max_img_seq_len, :])
            img_len = img_feat.shape[0]
            img_padding_len = 0
        else:
            img_padding_len = self.max_img_seq_len - img_len
            padding_matrix = torch.zeros((img_padding_len, img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # generate attention_mask
        att_mask_type = self.args.att_mask_type
        if att_mask_type == "CLR":
            attention_mask = [1] * seq_len + [0] * seq_padding_len + \
                             [1] * img_len + [0] * img_padding_len
        else:
            # use 2D mask to represent the attention
            max_len = self.max_seq_len + self.max_img_seq_len
            attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
            # full attention of C-C, L-L, R-R
            c_start, c_end = 0, seq_a_len
            l_start, l_end = seq_a_len, seq_len
            r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
            attention_mask[c_start : c_end, c_start : c_end] = 1
            attention_mask[l_start : l_end, l_start : l_end] = 1
            attention_mask[r_start : r_end, r_start : r_end] = 1
            if att_mask_type == 'CL':
                attention_mask[c_start : c_end, l_start : l_end] = 1
                attention_mask[l_start : l_end, c_start : c_end] = 1
            elif att_mask_type == 'CR':
                attention_mask[c_start : c_end, r_start : r_end] = 1
                attention_mask[r_start : r_end, c_start : c_end] = 1
            elif att_mask_type == 'LR':
                attention_mask[l_start : l_end, r_start : r_end] = 1
                attention_mask[r_start : r_end, l_start : l_end] = 1
            else:
                raise ValueError("Unsupported attention mask type {}".format(att_mask_type))
         
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        return (input_ids, attention_mask, segment_ids, img_feat,label_final)

    def tensorize_example(self, text_a, img_feat,text_c=None, text_d=None,text_b=None,dialog_id=None,label_final=None,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):
        #print(text_a)
        self.max_seq_len = self.args.max_seq_length
        self.max_seq_a_len = self.args.max_seq_length
        #print("max_seq_length", self.args.max_seq_length,self.max_seq_len)
        #if self.is_train:
        tokens_a = self.tokenizer.tokenize(text_a)
        
        if len(tokens_a) > self.args.max_seq_length - 2:
            tokens_a = tokens_a[:(self.args.max_seq_length - 2)]
        #print(len(tokens_a))

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        
        #print(len(tokens))
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len = len(tokens)
        if 1:
            # pad text_a to keep it in fixed length for better inference.
           # padding_a_len = self.max_seq_a_len - seq_a_len
           # tokens += [self.tokenizer.pad_token] * padding_a_len
           # segment_ids += ([pad_token_segment_id] * padding_a_len)
            #print("text_d type",type(text_d))
            #print("text_c type",type(text_c))
            #print("text_b type",type(text_b))

            tokens_c = self.tokenizer.tokenize(text_c)
            tokens_d = self.tokenizer.tokenize(text_d)
       
            tokens_b = self.tokenizer.tokenize(text_b)

            if len(tokens_c) > self.args.max_seq_length - len(tokens) - 3 - len(tokens_b)  - len(tokens_d):
                  #print("len  tokens_c",len(tokens_c),len(tokens),len(tokens_b),len(tokens_d))
                  tokens_c = tokens_c[-(self.args.max_seq_length - len(tokens) - 3 - len(tokens_b) - len(tokens_d)): ]                 
                  #print("len  tokens_c after",len(tokens_c))
            tokens += tokens_c + [self.tokenizer.sep_token]
            

            tokens += tokens_d + [self.tokenizer.sep_token]
            

            if len(tokens_b) > self.args.max_seq_length - len(tokens) - 1:
                tokens_b = tokens_b[: (self.args.max_seq_length - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + len(tokens_c) + len(tokens_d)  + 3)



        seq_len = len(tokens)
        #print("seq_len",seq_len)
        seq_padding_len = self.max_seq_len - seq_len
        #print("seq_padding_len",seq_padding_len)

        tokens += [self.tokenizer.pad_token] * seq_padding_len
        segment_ids += [pad_token_segment_id] * seq_padding_len
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        #print("input_ids shape",len(input_ids))
       
        # image features
        img_len = img_feat.shape[0]
        img_feat= torch.tensor(img_feat, dtype=torch.float32)
        label_final = torch.tensor(label_final, dtype=torch.int64)
       
            
        #padding_matrix = torch.zeros((self.args.max_seq_length+1))    
        #label_feat =     torch.cat((padding_matrix,label_feat), 0) 
        
        if img_len > self.max_img_seq_len:
            img_feat = torch.tensor(img_feat[0 : self.max_img_seq_len, :])
            img_len = img_feat.shape[0]
            img_padding_len = 0
        else:
            img_padding_len = self.max_img_seq_len - img_len
            padding_matrix = torch.zeros((img_padding_len, img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)
       
        # generate attention_mask
        att_mask_type = self.args.att_mask_type
        if att_mask_type == "CLR":
            attention_mask = [1] * seq_len + [0] * seq_padding_len + \
                             [1] * img_len + [0] * img_padding_len
        else:
            # use 2D mask to represent the attention
            max_len = self.max_seq_len + self.max_img_seq_len
            attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
            # full attention of C-C, L-L, R-R
            c_start, c_end = 0, seq_a_len
            l_start, l_end = seq_a_len, seq_len
            r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
            attention_mask[c_start : c_end, c_start : c_end] = 1
            attention_mask[l_start : l_end, l_start : l_end] = 1
            attention_mask[r_start : r_end, r_start : r_end] = 1
            if att_mask_type == 'CL':
                attention_mask[c_start : c_end, l_start : l_end] = 1
                attention_mask[l_start : l_end, c_start : c_end] = 1
            elif att_mask_type == 'CR':
                attention_mask[c_start : c_end, r_start : r_end] = 1
                attention_mask[r_start : r_end, c_start : c_end] = 1
            elif att_mask_type == 'LR':
                attention_mask[l_start : l_end, r_start : r_end] = 1
                attention_mask[r_start : r_end, l_start : l_end] = 1
            else:
                raise ValueError("Unsupported attention mask type {}".format(att_mask_type))
         
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        #print("input_ids shape",input_ids.shape)
        
        #print("img_feat shape",img_feat.shape)

        if self.is_train:
            #didout = torch.tensor(label_final, dtype=torch.int64)
            #id_final_out = torch.tensor(label_final, dtype=torch.int64)
            #print("input_ids",input_ids.shape)
            #print("attention_mask",attention_mask.shape)
            #print("segment_ids",segment_ids.shape)
            #print("img_feat",img_feat.shape)
            #print("label_final",label_final.shape)
       
            return (input_ids, attention_mask, segment_ids, img_feat,label_final)
       
        else:
            didout = torch.tensor(dialog_id, dtype=torch.int64)
           # id_final_out = torch.tensor(id_final, dtype=torch.int64)

        return (input_ids, attention_mask, segment_ids, img_feat,label_final,didout)





        
    def get_one_item(self,index):
        if self.is_train:
             
             dialogue_final, scene_thisround,dialog_id, id_final, type_final, bbox_final, slotvalue_final, image_feature,label_final = self.feature[index]
        else:
             dialogue_final, scene_thisround,dialog_id, id_final, type_final, bbox_final, slotvalue_final, image_feature,label_final = self.devfeature[index]

      
        textc_list = dialogue_final[:-1]
        textc = ''
        for text in textc_list:
             textc = textc + text

        caption_list = dialogue_final[-1:]
        
        #print(caption_list)
        caption = ''
        for text in caption_list:
             caption = caption + text
        #print(caption)
        
        textd_list = slotvalue_final
        if len(textd_list) <= 1:
            textd = ''
        else:
            textd = "first:"
            
        num = 0
        #print(textb_list) 
        for text in textd_list:
             #print("type",type(text))
             num = num + 1
             if textd == '':
                 textd = text
             else:
                 if num == 1:
                     textd = "first:"  + text                    
                 if num == 2:
                    textd = textd + ' second:' + text
                 if num == 3:
                    textd = textd + ' third:' + text
                 if num == 4:
                     textd = textd + ' fourth:' + text
                 if num == 5:
                      textd = textd + ' fifth:' + text
                 if num == 6:
                      textd = textd + ' sixth:' + text
                 if num == 7:
                      textd = textd + ' seventh:' + text
                 if num == 8:
                      textd = textd + ' eighth:' + text
                 if num == 9:
                      textd = textd + ' ninth:' + text
                 if num == 10:
                      textd = textd + ' tenth:' + text                     

        #print(texta)
        textb = ' '
        #textb_list = type_final
        #print(textb_list) 
        #for text in textb_list:
        #     if textb == ' ':
        #         textb = text
         #    else:
         #        textb = textb + ' ' + text

        #print(textb)
        #print(image_feature)
        image_feature = np.array(image_feature)
        #print(image_feature.shape)
        #print(label_final)
        return caption,str(textb),str(textc),str(textd),image_feature,dialog_id,label_final

    def __getitem__(self, idx):
        #img_idx = self.get_image_index(idx)
        #img_key = self.image_keys[img_idx]
        #features = self.get_image_features(img_idx)
        #caption = self.get_caption(idx)
        #od_labels = self.get_od_labels(img_idx)
        #textc history textb type textd slot value
        caption,textb,textc,textd,image_feature,dialog_id,label_final = self.get_one_item(idx)
        #print("caption",caption)
        #print("textb",textb)
        #print("textc",textc)
        #print("textd",textd)
       # #print("image_feature",image_feature.shape)
        

        image_feature= torch.tensor(image_feature, dtype=torch.float32)
        #print(image_feature.shape)
        #if self.is_train:
        #     caption = caption
        #else:
        #     caption = ""
            
        #example = self.tensorize_example(texta, image_feature, label_final=label_final, text_b=textb)        
        example = self.tensorize_example(caption, image_feature,textc, textd,textb,dialog_id,label_final)
     #   print("example",example)
        return idx, example


    def get_image(self, image_id):
        image_idx = self.image_id2idx[str(image_id)]
        row = self.img_tsv.seek(image_idx)
        num_boxes = int(row[1])
        features = np.frombuffer(base64.b64decode(row[-1]),
                                 dtype=np.float32).reshape((num_boxes, -1))
        t_features = torch.from_numpy(features)
        return t_features

    def __len__(self):
        if not self.is_train:
            return len(self.devfeature)  
        return   len(self.feature)


def compute_score_with_logits_test(logits, labels,flag=0):
    #logits_new = logits.cpu()


    pred = torch.argmax(logits, dim=1)
    
    if  1:
       #print("=logit label=============")
       #print(logits)
       #print(labels)
       #print("==============")
       labels_new = labels.cpu()
       pred_new = pred.cpu()
       logits_new = logits.cpu()
       logits_new = logits_new

       result = metrics.classification_report(labels_new,pred_new,target_names=None,digits=4)
       print("one result",result)
       print("logits_newshape",logits_new.shape)
    acc = 0
    total = 0
    sus = 0
    for i in range(len(labels)):
        if labels[i] == pred[i]:
           sus = sus +1
        total = total + 1
    acc = sus/total    
    return  acc,labels_new,pred_new,logits_new


def compute_score_with_logits(logits, labels,flag=0):
    #logits_new = logits.cpu()


    pred = torch.argmax(logits, dim=1)
    #if 0:
    if  flag == 0:
       #print("=logit label=============")
       #print(logits)
       #print(labels)
       #print("==============")
       labels_new = labels.cpu()
       pred_new = pred.cpu()
       result = metrics.classification_report(labels_new,pred_new,target_names=None,digits=4)
       print("new",result)
    acc = 0
    total = 0
    sus = 0
    for i in range(len(labels)):
        if labels[i] == pred[i]:
           sus = sus +1
        total = total + 1
    acc = sus/total    
    return  acc


def compute_score_with_logits_old(logits, labels):
    scores = torch.zeros_like(labels).cuda()
    error_a = 0
    total_a = 0
    total = 0
    error = 0

    for i, (logit_l, label_l) in enumerate(zip(logits, labels)):
          total_a = total_a + 1
          error_one = 0
          for i, (logit, label) in enumerate(zip(logit_l, label_l)):
            logit_ = torch.sigmoid(logit)
            
            total = total + 1
            if (logit_ >= 0.5 and label == 1) or (logit_ < 0.5 and label == 0):
                pass
            else:
                error = error + 1
                error_one = error_one  + 1
          if error_one > 0 :
              error_a = error_a + 1
    return float(100-(100*error_a)/total_a),float(100-(100*error)/total),error_a,total_a


def compute_ranks(dataset, results):
    labels = np.array([dataset.get_label(i) for i in range(len(dataset))])
    similarities = np.array([results[i] for i in range(len(dataset))])
    if dataset.has_caption_indexs:
        num_captions_per_img = dataset.num_captions_per_img
    else:
        num_captions_per_img = len(dataset.img_keys) * dataset.num_captions_per_img
    labels = np.reshape(labels, [-1, num_captions_per_img])
    similarities = np.reshape(similarities, [-1, num_captions_per_img])
    i2t_ranks, t2i_ranks = [], []
    for lab, sim in zip(labels, similarities):
        inds = np.argsort(sim)[::-1]
        rank = num_captions_per_img
        for r, ind in enumerate(inds):
            if lab[ind] == 1:
                rank = r
                break
        i2t_ranks.append(rank)
    if not dataset.has_caption_indexs:
        labels = np.swapaxes(labels, 0, 1)
        similarities = np.swapaxes(similarities, 0, 1)
        for lab, sim in zip(labels, similarities):
            inds = np.argsort(sim)[::-1]
            rank = num_captions_per_img
            for r, ind in enumerate(inds):
                if lab[ind] == 1:
                    rank = r
                    break
            t2i_ranks.append(rank)
    return i2t_ranks, t2i_ranks


def save_checkpoint(model, tokenizer, args, epoch, global_step):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, global_step))
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    save_num = 0
    while (save_num < 10):
        try:
            model_to_save.save_pretrained(checkpoint_dir)
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            save_num += 1
    if save_num == 10:
        logger.info("Failed to save checkpoint after 10 trails.")
    return


def train(args, train_dataset, val_dataset, model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
            batch_size=args.train_batch_size, num_workers=args.num_workers)
    #for i in range(10):
       #train_dataset.__getitem__(i)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // \
                args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps \
                * args.num_train_epochs

    # Prepare optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not \
            any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if \
            any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.scheduler == "constant":
        scheduler = WarmupConstantSchedule(
                optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = WarmupLinearSchedule(
                optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step, global_loss, global_acc =0,  0.0, 0.0
    model.zero_grad()
    log_json = []
    best_score = 0
    for epoch in range(int(args.num_train_epochs)):
        for step, (_, batch) in enumerate(train_dataloader):
   # for i,batch in train_dataloader:
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                'input_ids':      batch[0]  ,
                'attention_mask':  batch[1] ,
                'token_type_ids': batch[2],
                'img_feats':      batch[3] ,
                'labels':       batch[4] 
            }
            outputs = model(**inputs)
            loss, logits = outputs[:2]
            if args.n_gpu > 1: 
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            batch_acc = compute_score_with_logits(logits, inputs['labels'],1)
          #  batch_acc = batch_score.item() / (args.train_batch_size * 2)
            global_loss += loss.item()
            global_acc += batch_acc
           # batch_acc = global_acc
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                if global_step % args.logging_steps == 0:
                    
                    batch_acc = compute_score_with_logits(logits, inputs['labels'],0)
                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f}), " \
                        "score: {:.4f} ({:.4f})".format(epoch, global_step, 
                        optimizer.param_groups[0]["lr"], loss, global_loss / global_step, 
                        batch_acc, global_acc / global_step)
                    )

                if (args.save_steps > 0 and global_step % args.save_steps == 0) or \
                        global_step == t_total:
                    save_checkpoint(model, tokenizer, args, epoch, global_step) 
                    # evaluation
                    if args.evaluate_during_training: 
                        logger.info("Perform evaluation at step: %d" % (global_step))
                        test_result = test(args, model, val_dataset)
                        #eval_result = evaluate(val_dataset, test_result)
                        #rank_accs = eval_result['i2t_retrieval']
                        #if rank_accs['R@1'] > best_score:
                        #    best_score = rank_accs['R@1']
                        #poch_log = {'epoch': epoch, 'global_step': global_step, 
                        #             'R1': rank_accs['R@1'], 'R5': rank_accs['R@5'], 
                        #             'R10': rank_accs['R@10'], 'best_R1':best_score}
                        #log_json.append(epoch_log)

    return global_step, global_loss / global_step


def test(args, model, eval_dataset):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
            batch_size=args.eval_batch_size, num_workers=args.num_workers)
    #for i in range(10):
    #   eval_dataset.__getitem__(i)    
    logger.info("Num examples = {}".format(len(eval_dataset)))
    logger.info("Evaluation batch size = {}".format(args.eval_batch_size))
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()
    results = {}
    error = 0
    total = 0
    softmax = nn.Softmax(dim=1)
    #for step, (_, batch) in enumerate(eval_dataloader):
    labels = []
    preds = []
    dids = []
    ids = []

    logitsoutput = []
        #model.eval()
    for indexs, batch in tqdm(eval_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        
        with torch.no_grad():
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'img_feats':      batch[3],
                'labels':         batch[4]
            }
            didsinput = batch[5]

            #print(inputs)
            _, logits = model(**inputs)[:2]
            if args.num_labels == 2:
                probs = softmax(logits)
                result = probs[:, 1] # the confidence to be a matched pair
            else:
                result = logits
            acc,labels_new,pred_new,logitnew  = compute_score_with_logits_test(logits, inputs['labels'],0) 
            for label in labels_new:
                labels.append(label)
            for pred in pred_new:
                preds.append(pred)
            for logitoutput in logitnew:
                logitsoutput.append(logitoutput)
            print(didsinput)
            for did in didsinput.cpu():
                #print(did)
                dids.append(did)                
     

            print('================acc',acc)
    try:
       result = metrics.classification_report(labels,preds,target_names=None,digits=4)
    except:
       pass
    allresult = []
    print(len(dids),len(preds),len(labels),len(logitsoutput))
    for i in range(len(dids)):
        allresult.append((dids[i],preds[i],labels[i],logitsoutput[i]))
    print("total",result)            
    print("total ================================") 
    if args.eval_model_dir:
       filename =  args.eval_model_dir + "/resultfromsys.txt"
    else:
       filename = args.output_dir + "/resultfromsys.txt"
    with open(filename, 'w') as fp:
        fp.writelines(result)
    picklebinfilename = args.eval_model_dir + "/resultfromsys.bin"
    print("picklebinfilename",picklebinfilename)
    with open(picklebinfilename,'wb') as fp1:
             pickle.dump(allresult,fp1) 
             print("picklebinfilename",picklebinfilename)

             #result = [_.to(torch.device("cpu")) for _ in result]
            #results.update({idx.item(): res.item() for idx, res in zip(indexs, result)})


def evaluate(eval_dataset, test_results):
    i2t_ranks, t2i_ranks = compute_ranks(eval_dataset, test_results)
    rank = [1, 5, 10]
    i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]
    logger.info("I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                i2t_accs[0], i2t_accs[1], i2t_accs[2]))
    eval_result = {"i2t_retrieval": {"R@1": i2t_accs[0], "R@5": i2t_accs[1], "R@10": i2t_accs[2]}}
    if t2i_ranks:
        t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]
        logger.info("T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                    t2i_accs[0], t2i_accs[1], t2i_accs[2]))
        eval_result["t2i_retrieval"] = {"R@1": t2i_accs[0], "R@5": t2i_accs[1], "R@10": t2i_accs[2]}
    return eval_result


def get_predict_file(args):
    cc = []
    data = op.basename(op.join(args.data_dir, '')[:-1])
    if data != 'coco_ir':
        cc.append(data)
    cc.append(args.test_split)
    if args.add_od_labels:
        cc.append('wlabels{}'.format(args.od_label_type))
    return op.join(args.eval_model_dir, '{}.results.pt'.format('.'.join(cc))) 


def restore_training_settings(args):
    assert not args.do_train and (args.do_test or args.do_eval)
    train_args = torch.load(op.join(args.eval_model_dir, 'training_args.bin'))
    override_params = ['do_lower_case', 'img_feature_type', 'max_seq_length', 
            'max_img_seq_length', 'add_od_labels', 'od_label_type',
            'use_img_layernorm', 'img_layer_norm_eps']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(param,
                    test_v, train_v))
                setattr(args, param, train_v)
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='datasets/coco_ir', type=str, required=False,
                        help="The input data dir with all required files.")
    parser.add_argument("--img_feat_file", default='datasets/coco_ir/features.tsv', type=str, required=False,
                        help="The absolute address of the image feature file.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model or model type. required for training.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--loss_type", default='sfmx', type=str, 
                        help="Loss function types: support kl, sfmx")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str, 
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--max_seq_length", default=450, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, "
                             "sequences shorter will be padded."
                             "This number is calculated on COCO dataset" 
                             "If add object detection labels, the suggested length should be 70.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run performance valuation."
                       "do not activate if we want to inference on dataset without gt labels.")
    parser.add_argument("--test_split", default='test', type=str, help='data split name.')
    parser.add_argument("--eval_img_keys_file", default='', type=str, 
                        help="image key tsv to select a subset of images for evaluation. "
                        "This is useful in 5-folds evaluation. The topn index file is not " 
                        "needed in this case.")
    parser.add_argument("--eval_caption_index_file", default='', type=str, 
                        help="index of a list of (img_key, cap_idx) for each image."
                        "this is used to perform re-rank using hard negative samples."
                        "useful for validation set to monitor the performance during training.")
    parser.add_argument("--cross_image_eval", action='store_true', 
                        help="perform cross image inference, ie. each image with all texts from other images.")
    parser.add_argument("--add_od_labels", default=False, action='store_true', 
                        help="Whether to add object detection labels or not.")
    parser.add_argument("--od_label_type", default='vg', type=str, 
                        help="label type, support vg, gt, oid")
    parser.add_argument("--att_mask_type", default='CLR', type=str, 
                        help="attention mask type, support ['CL', 'CR', 'LR', 'CLR']"
                        "C: caption, L: labels, R: image regions; CLR is full attention by default."
                        "CL means attention between caption and labels."
                        "please pay attention to the order CLR, which is the default concat order.")
    parser.add_argument("--do_lower_case", action='store_true', 
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    parser.add_argument("--max_img_seq_length", default=10, type=int, 
                        help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int, 
                        help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='frcnn', type=str,
                        help="Image feature type.")
    parser.add_argument("--use_img_layernorm", type=int, default=1,
                        help="Normalize image features with bertlayernorm")
    parser.add_argument("--img_layer_norm_eps", default=1e-12, type=float,
                        help="The eps in image feature laynorm layer")
    parser.add_argument("--per_gpu_train_batch_size", default=12, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=12, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--output_mode", default='classification', type=str,
                        help="output mode, support classification or regression.")
    parser.add_argument("--num_labels", default=2, type=int, 
                        help="num_labels is 2 for classification and 1 for regression.")
    parser.add_argument("--classoneweight", default=0.5, type=float, 
                        help="num_labels is 2 for classification and 1 for regression.")
    parser.add_argument("--num_captions_per_img_train", default=5, type=int,
                        help="number of positive matched captions for each training image.")
    parser.add_argument("--num_captions_per_img_val", default=5, type=int,
                        help="number of captions for each testing image.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear.")
    parser.add_argument("--num_workers", default=4, type=int, help="Workers in dataloader.")
    parser.add_argument("--num_train_epochs", default=10, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, 
                        help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=-1, 
                        help="Save checkpoint every X steps. Will also perform evaluatin.")
    parser.add_argument("--evaluate_during_training", action='store_true', 
                        help="Run evaluation during training at each save_steps.")
    parser.add_argument("--eval_model_dir", type=str, default='', 
                        help="Model directory for evaluation.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA.")
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")
    args = parser.parse_args()

    global logger
    mkdir(args.output_dir)
    logger = setup_logger("vlpretrain", args.output_dir, 0)

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args.seed, args.n_gpu)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.n_gpu)
    logger.info('output_mode: {}, #Labels: {}'.format(args.output_mode, args.num_labels))
 
    config_class, tokenizer_class = BertConfig, BertTokenizer
    model_class = ImageBertForSequenceClassification
    if args.do_train:
        config = config_class.from_pretrained(args.config_name if args.config_name else \
            args.model_name_or_path, num_labels=args.num_labels, finetuning_task='ir')
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
            else args.model_name_or_path, do_lower_case=args.do_lower_case)
        config.img_feature_dim = args.img_feature_dim
        config.img_feature_type = args.img_feature_type
        config.hidden_dropout_prob = args.drop_out
        config.loss_type = args.loss_type
        config.classoneweight = args.classoneweight

        config.img_layer_norm_eps = args.img_layer_norm_eps
        config.use_img_layernorm = args.use_img_layernorm
        model = model_class.from_pretrained(args.model_name_or_path, 
            from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    else:
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        config = config_class.from_pretrained(checkpoint)
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model = model_class.from_pretrained(checkpoint, config=config)

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    if args.do_train:
        train_dataset = RetrievalDataset(tokenizer, args, 'train', is_train=True)
        val_dataset = RetrievalDataset(tokenizer, args, 'minival', is_train=False)

        global_step, avg_loss = train(args, train_dataset, val_dataset, model, tokenizer)
        logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)
        

    if args.do_eval:
        val_dataset = RetrievalDataset(tokenizer, args, 'minival', is_train=False)

        #checkpoint = args.eval_model_dir
        #assert op.isdir(checkpoint)
        #logger.info("Evaluate the following checkpoint: %s", checkpoint)
        #model = model_class.from_pretrained(checkpoint, config=config)        
        test_result = test(args, model, val_dataset)
    return
    # inference and evaluation
    if args.do_test or args.do_eval:
        #args = restore_training_settings(args)
        test_dataset = RetrievalDataset(tokenizer, args, args.test_split, is_train=False)
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model = model_class.from_pretrained(checkpoint, config=config)
        model.to(args.device)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        pred_file = get_predict_file(args)
        if op.isfile(pred_file):
            logger.info("Prediction file exist, skip inference.")
            if args.do_eval:
                test_result = torch.load(pred_file)
        else:
            test_result = test(args, model, test_dataset)
            torch.save(test_result, pred_file)
            logger.info("Prediction results saved to {}.".format(pred_file))

        if args.do_eval:
            eval_result = evaluate(test_dataset, test_result)
            result_file = op.splitext(pred_file)[0] + '.eval.json'
            with open(result_file, 'w') as f:
                json.dump(eval_result, f)
            logger.info("Evaluation results saved to {}.".format(result_file))


if __name__ == "__main__":
    main()
