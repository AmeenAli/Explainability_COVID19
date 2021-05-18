import os
import sys
import numpy as np
from math import floor
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import cv2

import torchvision
from torchvision import models
import pdb

class ResNet(nn.Module):

    def __init__(self,conf):
        super(ResNet, self).__init__()
        basenet = eval('models.'+'resnet50')(pretrained=conf.pretrained)
        self.conv3 = nn.Sequential(*list(basenet.children())[:-4])
        self.conv4 = list(basenet.children())[-4]
        self.midlevel = False
        self.isdetach = True

        self.bn0 = nn.Sequential(nn.BatchNorm2d(512), nn.ReLU())
        self.bn1 = nn.Sequential(nn.BatchNorm2d(2048), nn.ReLU())

        if 'midlevel' in conf:
            self.midlevel = conf.midlevel
        if 'isdetach' in conf:
            self.isdetach = isdetacjh

        mid_dim = 1024
        feadim = 2048
        if conf.netname in ['resnet18','resnet34']:
            mid_dim = 256
            feadim = 512

        if self.midlevel:
            self.mcls = nn.Linear(mid_dim, conf.num_class)
            self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
            self.conv4_1 = nn.Sequential(nn.Conv2d(mid_dim, mid_dim, 1, 1), nn.ReLU() , nn.BatchNorm2d(1024))
        self.conv5 = list(basenet.children())[-3]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(feadim, 3)

    def set_detach(self,isdetach=True):
        self.isdetach = isdetach

    def forward(self, x):
        x = self.conv3(x)
        x = self.bn0(x)
        conv4 = self.conv4(x)
        x = self.conv5(conv4)
        x = self.bn1(x)
        patches = x
        fea_pool = self.avg_pool(x).view(x.size(0), -1)

        logits = self.classifier(fea_pool)

        if self.midlevel:
            if self.isdetach:
                conv4_1 = conv4.detach()
            else:
                conv4_1 = conv4
            conv4_1 = self.conv4_1(conv4_1)
            pool4_1 = self.max_pool(conv4_1).view(conv4_1.size(0),-1)
            mlogits = self.mcls(pool4_1)
        else:
            mlogits = None

        return logits,x.detach(),mlogits , patches


    def _init_weight(self, block):
        for m in block.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_params(self, param_name):
        ftlayer_params = list(self.conv3.parameters()) +\
                           list(self.conv4.parameters()) +\
                           list(self.conv5.parameters())
        ftlayer_params_ids = list(map(id, ftlayer_params))
        freshlayer_params = filter(lambda p: id(p) not in ftlayer_params_ids, self.parameters())

        return eval(param_name+'_params')


def get_net(conf):
    return ResNet(conf)
