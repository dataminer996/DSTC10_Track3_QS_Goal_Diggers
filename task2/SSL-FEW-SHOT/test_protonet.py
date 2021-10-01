import argparse
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from feat.dataloader.samplers import CategoriesSampler
from feat.models.protonet import ProtoNet
from feat.utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, compute_confidence_interval
from tensorboardX import SummaryWriter
from torch.utils.data import  SequentialSampler
from tqdm import tqdm
import os
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--model_type', type=str, default='AmdimNet', choices=['ConvNet', 'ResNet', 'AmdimNet'])
    parser.add_argument('--split_path', type=str, default='data/miniimagenet/split')
    parser.add_argument('--image_path', type=str, default='data/miniimagenet/0923_allimage')

    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'CUB', 'TieredImageNet'])    
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--gpu', default='0')

    ### AMDIM MODEL
    parser.add_argument('--ndf', type=int, default=256)
    parser.add_argument('--rkhs', type=int, default=2048)
    parser.add_argument('--nd', type=int, default=10)

    args = parser.parse_args()
    args.temperature = 1 # we set temperature = 1 during test since it does not influence the results
    pprint(vars(args))

    print(torch.cuda.is_available())
    set_gpu(args.gpu)
    
    if args.dataset == 'MiniImageNet':
        from feat.dataloader.mini_imagenet import MiniImageNet_Test as Dataset
    elif args.dataset == 'CUB':
        from feat.dataloader.cub import CUB as Dataset
    elif args.dataset == 'TieredImageNet':
        from feat.dataloader.tiered_imagenet import tieredImageNet as Dataset       
    else:
        raise ValueError('Non-supported Dataset.')

    model = ProtoNet(args)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()    
    test_set = Dataset('val', args)    
    eval_sampler = SequentialSampler(test_set)   ##tangliang
  #  sampler = CategoriesSampler(test_set.label, 10000, args.way, args.shot + args.query)
    loader = DataLoader(test_set, batch_sampler=eval_sampler,batch_size=1, num_workers=8)
  #  loader = DataLoader(test_set, batch_sampler=eval_sampler,batch_size=1, num_workers=8, pin_memory=True)

    test_acc_record = np.zeros((100000,))

    model.load_state_dict(torch.load(args.model_path)['params'])
    model.eval()

    ave_acc = Averager()
    label = torch.arange(args.way).repeat(args.query)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)
    output_result = []
    device = torch.device("cuda")    
    
    with torch.no_grad():
    
        #for i, batch in enumerate(loader, 1):
       # for i, batch in tqdm(loader):
        for i,batch in test_set:
            #batch = tuple(t.to(device) for t in batch)
           # print(batch)
           # if torch.cuda.is_available():
           #     data, _ = [_.cuda() for _ in batch]
           # else:
           #     data = batch[0]
            data_query =  batch[0].cuda()
            data_shot =  batch[1].cuda()
            label_index = batch[2].cuda()
            #imagename = batch[3]

          #  k = args.way * args.shot
          #  data_shot, data_query = data[:k], data[k:]
            print(data_shot.shape)
            print(data_query.shape)
            #print("label",label_index.shape)
            print("label",label_index)

            logits = model(data_shot, data_query)
            #print("logits",logits)
            acc = count_acc(logits, label_index)
            pred = torch.argmax(logits, dim=1)
            print("pred",pred)

            ave_acc.add(acc)
            test_acc_record[i-1] = acc
            print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
            output_result.append((i,pred.cpu(),label_index.cpu(),logits.cpu()))
            
    bin_path = osp.join(os.path.dirname(args.model_path), "result" + '.bin')
    with open(bin_path,'wb') as fp:
                 pickle.dump(output_result,fp)            
    m, pm = compute_confidence_interval(test_acc_record)
    print('Test Acc {:.4f} + {:.4f}'.format(m, pm))

