import warnings 
warnings.filterwarnings('ignore')
import numpy as np
import os
import torch.nn as nn
import torch
import argparse
import random
import time
from torchvision.transforms import Compose, ToTensor, RandomCrop
from metrics.miou import mIOUMetrics
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from models.segformer.segformer import Seg
from metrics.compute_iou import fast_hist, per_class_iu

from dataset.densepass_val_dataset import densepass_val
from dataset.city.City_dataset import CityDataset
from dataset.densepass_train_dataset import densepass_train
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith('module.') else key  # Remove 'module.' prefix
        new_state_dict[new_key] = value
    return new_state_dict

NAME_CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                'traffic light', 'traffic sign',
                'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                'truck', 'bus', 'train', 'motorcycle', 'bicycle']

class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long()
        
        #return np.asarray(image, np.float32)

def main():
    parser = argparse.ArgumentParser(description='pytorch implemention')
    parser.add_argument('--batch-size', type=int, default=6, metavar='N',
                        help='input batch size for training (default: 6)')
    parser.add_argument('--iterations', type=int, default=200000, metavar='N',
                        help='number of epochs to train (default: 30000)')
    parser.add_argument('--lr', type=float, default=6e-5, metavar='LR',
                        help='learning rate (default: 6e-5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()

    best_performance_pin = 0.0
    best_performance_pan = 0.0

    save_path = "{}{}".format(args.save_root,args.exp_name)

    torch.cuda.set_device(args.local_rank)
    with torch.cuda.device(args.local_rank):
        dist.init_process_group(backend='nccl',init_method='env://') #nccl
        if dist.get_rank() == 0:
            print(args)
            print('init lr: {}, batch size: {}, gpus:{}'.format(args.lr, args.batch_size, dist.get_world_size()))

        num_classes = 19

        # DensePASS dataset
        # ------------------------------------------------------------------------------------------------------------#
        input_transform_cityscapes = Compose([ToTensor(),])
        target_transform_cityscapes = Compose([ToLabel(),])  
        val_root = "/DensePASS"  
        val_DensePASS = densepass_val(val_root, input_transform=input_transform_cityscapes,target_transform=target_transform_cityscapes, target=True)
        pass_val_loader = torch.utils.data.DataLoader(val_DensePASS,batch_size=1,shuffle=False,num_workers=12)
        model = Seg(backbone='mit_b2',num_classes=num_classes,embedding_dim=512,pretrained=True) 
        pretrained_dict = torch.load('/city_b2_52.99.pth', map_location='cpu')
        pretrained_dict = remove_module_prefix(pretrained_dict)

        model.load_state_dict(pretrained_dict)
        
        model = model.to(args.local_rank)

        with torch.no_grad():     
            model.eval()
            print('Panoramic Dataset')
            best_performance_pan = validation(num_classes, NAME_CLASSES, args.local_rank, pass_val_loader, model)
            
            model.train()

def validation(num_classes, NAME_CLASSES, device, testloader, model1):
    hist = np.zeros((num_classes, num_classes))

    for i, data in enumerate(testloader):

        [image, label, _, _] = data 
        image, label = image.to(device), label.to(device)
        
        with torch.no_grad():
            output, _ = model1(image)
        output = torch.argmax(output, 1).squeeze(0).cpu().data.numpy()

        label = label.cpu().data[0].numpy()
        hist += fast_hist(label.flatten(), output.flatten(), num_classes)

    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print('===>{:<15}:\t{}'.format(NAME_CLASSES[ind_class], str(round(mIoUs[ind_class] * 100, 2))))
    bestIoU = round(np.nanmean(mIoUs) * 100, 2)
    print('===> mIoU: ' + str(bestIoU))
      
    print('val_mIoU',bestIoU)
    return bestIoU

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    print('file name: ', __file__)
    setup_seed(1234)
    main()                   
