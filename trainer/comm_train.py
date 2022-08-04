import os
import sys
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import time
import logging
from utils import *
#from pytorch_metric_learning import losses as new_losses


class local_pos_loss(nn.Module):
    def __init__(self, conf):
        super(local_pos_loss, self).__init__()
        self.lsoftmax = nn.LogSoftmax(dim=1)

    def get_triplets(self, mask, n1=2, n2=2):
        index = torch.topk(mask.reshape(mask.shape[0], -1), n1, dim=1)[1][..., None]
        a = index // 7
        b = index % 7
        max_values = torch.cat([a, b], dim=2)
        index = torch.topk((-mask).reshape(mask.shape[0], -1), n2, dim=1)[1][..., None]
        a = index // 7
        b = index % 7
        min_values = torch.cat([a, b], dim=2)
        return max_values, min_values

    def latent_sample(self, z, p):
        bs = z.shape[0] # batchsize
        zSize = z.shape[1] # 2048
        nP = p.shape[2] # 4
        z = z.view(bs*zSize, z.shape[2], z.shape[3]).unsqueeze(dim=1) # [batchsize * 2048 , 1 , 7 , 7]
        p = p.repeat(1, zSize, 1, 1) # [batchsize , 2048 , 4 , 2]
        p = p.view(bs*zSize, p.shape[2], p.shape[3]).unsqueeze(dim=1) # [batchsize * 2048, 1, 4, 2]
        c = F.grid_sample(z, p).squeeze().view(bs, zSize, nP) # [batchsize , 2048 , 4]
        return c

    def forward(self, z, mask):
        positives, negatives = self.get_triplets(mask, n1=2, n2=2)
        
        # positives : [batchsize , 2 , 2] , negatives : [batchsize , 2 , 2]
        
        p = torch.cat([positives, negatives], dim=1)
        p = p.unsqueeze(1).float().cuda()
        
        # p : [batchsize , 1, 4, 2]
        
        a = self.latent_sample(z, p) # [batchsize , 2048 , 4]
        score_mat = a.permute(0, 2, 1) @ a # [batchsize , 4 , 2048] X [batchsize , 2048 , 4] = [bathsize , 4 , 4]
        
        loss_mat1 = -self.lsoftmax(score_mat[:, 0, 1:])[: , 0] # [batchsize , 3] : element 0 is u_1 x u_1, we want u_1 x u_2

        positives, negatives = self.get_triplets(mask, n1=self.n, n2=2)
        p = torch.cat([negatives, positives], dim=1)
        p = p.unsqueeze(1).float().cuda()
        a = self.latent_sample(z, p)
        score_mat = a.permute(0, 2, 1) @ a 
        loss_mat2 = -self.lsoftmax(score_mat[:, 0, 1:])[:, 0]

        loss_mat = loss_mat1 + loss_mat2
        return loss_mat.mean()



def get_maps(input , target, model , conf):
  wfmaps,_ = get_spm(input, target, conf, model)
  return wfmaps


def train(train_loader, model, criterion, optimizer, conf, wmodel=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cpc = AverageMeter()
    scores = AverageAccMeter()
    end = time.time()
    model.train()

    cpc_loss_local = local_pos_loss(conf)


    time_start = time.time()
    pbar = tqdm(train_loader, dynamic_ncols=True, total=len(train_loader))
    mixmethod = None
    clsw = None
    if 'mixmethod' in conf:
        if 'baseline' not in conf.mixmethod:
            mixmethod = conf.mixmethod
            if wmodel is None:
                wmodel = model

    for idx, (input, target , site) in enumerate(pbar):
        if idx > int(0.8 * len(train_loader)):
         break
        # measure data loading time
        data_time.add(time.time() - end)
        input = input.cuda()
        target = target.cuda()
        
        mask = get_maps(input, target, model, conf) # [batchsize , 224 , 224]
        mask = mask.unfold(1, 32, 32).unfold(2, 32, 32) # [batchsize , 7 , 7 , 32 , 32]
        mask = torch.sum(mask, dim=(3, 4)).unsqueeze(1) # [batchsize , 1 , 7 , 7]

        conf.mixmethod = 'snapmix'
        if 'baseline' not in conf.mixmethod:

            input, target_a, target_b, lam_a, lam_b = snapmix(input,target,conf,wmodel)
            output, a  , b , patches = model(input)
            # patches is of shape : [batchsize , 2048 , 7 , 7]
            cpc1 = cpc_loss_local(patches , mask)

            loss_a = criterion(output, target_a.cuda())
            loss_b = criterion(output, target_b.cuda())

            loss = torch.mean(loss_a* lam_a + loss_b* lam_b) + 0.1 * cpc1



            if 'inception' in conf.netname:
                loss1_a = criterion(moutput, target_a)
                loss1_b = criterion(moutput, target_b)
                loss1 = torch.mean(loss1_a* lam_a + loss1_b* lam_b)
                loss += 0.4*loss1

            if 'midlevel' in conf:
                if conf.midlevel:
                    loss_ma = criterion(moutput, target_a)
                    loss_mb = criterion(moutput, target_b)
                    loss += torch.mean(loss_ma* lam_a + loss_mb* lam_b)
        else:
            output,_,moutput = model(input)
            loss = torch.mean(criterion(output, target))

            if 'inception' in conf.netname:
                loss += 0.4*torch.mean(criterion(moutput,target))

            if 'midlevel' in conf:
                if conf.midlevel:
                    loss += torch.mean(criterion(moutput,target))

        # measure accuracy and record loss
        losses.add(loss, input.size(0))
        #losses.add(cont_loss , input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.add(time.time() - end)
        end = time.time()

        pbar.set_postfix(batch_time=batch_time.value(), data_time=data_time.value(), loss=losses.value(), score=0)

    return losses.value()
