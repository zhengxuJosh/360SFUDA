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

def plot_confusion_matrix(cm, class_names, save_path):
    """
    绘制混淆矩阵的热图，不包含数值标注，只显示颜色。
    :param cm: 混淆矩阵
    :param class_names: 类别名称列表
    :param save_path: 图片保存路径
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # 显示所有的刻度...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ...并且为它们标上类别名称
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # 旋转刻度标签并设置其对齐方式
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    # plt.show()

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

def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1-epoch/num_epochs)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

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
    parser.add_argument('--save_root', default = '',
                        help='Please add your model save directory') 
    parser.add_argument('--exp_name', default = '',
                        help='')
    parser.add_argument('--sup_set', type=str, default='train', help='supervised training set')
    parser.add_argument('--cutmix', default =False, help='cutmix')
    #================================hyper parameters================================#
    parser.add_argument('--alpha', type=float, default =0.5, help='alpha')
    parser.add_argument('--lamda', type=float, default =0.001, help='lamda')
    #================================================================================#
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
        # Cityscapes dataset
        # ------------------------------------------------------------------------------------------------------------#
        img_mean=[0.485, 0.456, 0.406]
        img_std=[0.229, 0.224, 0.225]
        city_crop_size = 512
        city_dataset_path = "/hpc2hdd/home/xzheng287/360Seg/TPAMI_360uda/dataset/city" # cityscapes dataset root
        city_label_dataset = CityDataset(f'{city_dataset_path}',split='train', base_size=2048, crop_size=city_crop_size, norm_mean=img_mean, norm_std=img_std, pass_like=False)        
        city_label_sampler = DistributedSampler(city_label_dataset, num_replicas=dist.get_world_size()) 
        city_label_loader = torch.utils.data.DataLoader(city_label_dataset,batch_size=args.batch_size,sampler=city_label_sampler,num_workers=12,worker_init_fn=lambda x: random.seed(time.time() + x),drop_last=True,)
        
        # city_val_dataset = CityDataset(f'{city_dataset_path}', split='val', base_size=2048, crop_size=city_crop_size, norm_mean=img_mean, norm_std=img_std)
        # val_loader = torch.utils.data.DataLoader(city_val_dataset,batch_size=args.batch_size,shuffle=False,num_workers=12)
        # DensePASS dataset
        # ------------------------------------------------------------------------------------------------------------#
        input_transform_cityscapes = Compose([ToTensor(),])
        target_transform_cityscapes = Compose([ToLabel(),])  
        train_root = "/hpc2hdd/home/xzheng287/360Seg/data" # training set root
        val_root = "/hpc2hdd/home/xzheng287/360Seg/data/DensePASS"   # validation set root
        train_DensePASS = densepass_train(train_root, list_path='/hpc2hdd/home/xzheng287/360Seg/TPAMI_360uda/dataset/train.txt',set=None)
        val_DensePASS = densepass_val(val_root, input_transform=input_transform_cityscapes,target_transform=target_transform_cityscapes, target=True)
        pass_train_sampler = DistributedSampler(train_DensePASS, num_replicas=dist.get_world_size())
        pass_train_loader = torch.utils.data.DataLoader(train_DensePASS,batch_size=args.batch_size,sampler=pass_train_sampler,num_workers=12,worker_init_fn=lambda x: random.seed(time.time() + x),drop_last=True,)
        pass_val_loader = torch.utils.data.DataLoader(val_DensePASS,batch_size=1,shuffle=False,num_workers=12)
        model = Seg(backbone='mit_b2',num_classes=num_classes,embedding_dim=512,pretrained=True) 
        pretrained_dict = torch.load('/hpc2hdd/home/xzheng287/360Seg/TPAMI_360uda/github/city_b2_52.99.pth', map_location='cpu')
        pretrained_dict = remove_module_prefix(pretrained_dict)

        model.load_state_dict(pretrained_dict)
        
        model = model.to(args.local_rank)
        # model = DistributedDataParallel(model,device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False, find_unused_parameters=True)
            
        # print(f'syn_val Dataset length:{len(syn_val)};')
        # print(f'dense_val Dataset length:{len(dense_val)};')

        with torch.no_grad():     
            model.eval()
            # print('Stanford Pinhole Dataset')
            # best_performance_pin = validation(num_classes, NAME_CLASSES, args.local_rank, stanford_syn_val_loader, model)
            
            print('Panoramic Dataset')
            best_performance_pan = validation(num_classes, NAME_CLASSES, args.local_rank, pass_val_loader, model)
            
            model.train()

def validation(num_classes, NAME_CLASSES, device, testloader, model1):
    hist = np.zeros((num_classes, num_classes))

    for i, data in enumerate(testloader):
        # if index % 100 == 0:
        #     print ('%d processd' % index)

        [image, label, _, _] = data 
        image, label = image.to(device), label.to(device)
        
        with torch.no_grad():
            output, _ = model1(image)
        output = torch.argmax(output, 1).squeeze(0).cpu().data.numpy()

        label = label.cpu().data[0].numpy()
        hist += fast_hist(label.flatten(), output.flatten(), num_classes)

    class_names = NAME_CLASSES
    plot_confusion_matrix(hist, class_names,
                          save_path="/hpc2hdd/home/xzheng287/360Seg/TPAMI_360uda/visualization/confusion_matrix/confusion_matrix.png")

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