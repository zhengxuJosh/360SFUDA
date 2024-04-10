import torch
import torch.nn as nn
import torch.nn.functional as F 
from .seghead import SegFormerHead
from . import MixT

class Seg(nn.Module):
    def __init__(self, backbone, num_classes=20, embedding_dim=256, pretrained=None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]

        ## initilize encoder
        if backbone == 'mit_b0':
            if pretrained:
                print("no pre")
                # state_dict = torch.load('/mit_b0.pth')
                # state_dict.pop('head.weight')
                # state_dict.pop('head.bias')
                # self.encoder.load_state_dict(state_dict,strict=False)
        if backbone == 'mit_b1':
            if pretrained:
                print("no pre")
                state_dict = torch.load('/mit_b1.pth')
                state_dict.pop('head.weight')
                state_dict.pop('head.bias')
                self.encoder.load_state_dict(state_dict,)
        if backbone == 'mit_b2':
            if pretrained:
                state_dict = torch.load('/mit_b2.pth')
                state_dict.pop('head.weight')
                state_dict.pop('head.bias')
                self.encoder.load_state_dict(state_dict,strict=False)
        self.backbone = backbone
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels, embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        
        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes, kernel_size=1, bias=False)

    def _forward_cam(self, x):
        
        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)
        
        return cam

    def get_param_groups(self):

        param_groups = [[], [], []] # 
        
        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):

            param_groups[2].append(param)
        
        param_groups[2].append(self.classifier.weight)

        return param_groups

    def forward(self, x):
        _, _, height, width = x.shape

        _x = self.encoder(x)

        feature =  self.decoder(_x)
        pred = F.interpolate(feature, size=(height,width), mode='bilinear', align_corners=False)

        return pred, _x[3]
