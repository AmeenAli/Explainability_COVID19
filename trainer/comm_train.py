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
from pytorch_metric_learning import losses as new_losses

class DeepLME(nn.Module):
    def __init__(self):
        super(DeepLME, self).__init__()
        self.lsoftmax = nn.LogSoftmax(dim=1)
    def latent_sample(self, z, p):
        bs = z.shape[0]
        zSize = z.shape[1]
        nP = p.shape[2]
        p = (p - 3.5) / 3.5
        z = z.contiguous().view(bs*zSize, z.shape[2], z.shape[3]).unsqueeze(dim=1)
        p = p.repeat(1, zSize, 1, 1)
        p = p.contiguous().view(bs*zSize, p.shape[2], p.shape[3]).unsqueeze(dim=1)
        c = F.grid_sample(z, p).squeeze().contiguous().view(bs, zSize, nP)
        return c
    def forward(self, z , p):
        p = p.unsqueeze(1).float().cuda()
        output = {}
        a = self.latent_sample(z, p)
        score_mat = a.permute(0, 2, 1) @ a
        loss_mat = -self.lsoftmax(score_mat[:, 0, 1:])[:,0] - self.lsoftmax(score_mat[:, -1, :-1])[:,-1]
        return loss_mat.mean()



def get_maps(input , target, model , conf):
  wfmaps,_ = get_spm(input,target,conf,model)
  return wfmaps

def get_triplets(mask):
   index = torch.topk(mask.reshape(mask.shape[0], -1), 2, dim=1)[1][..., None]
   a = index // 7
   b = index % 7
   max_values = torch.cat([a, b], dim=2)
   index = torch.topk((-mask).reshape(mask.shape[0], -1), 2, dim=1)[1][..., None]
   a = index // 7
   b = index % 7
   min_values = torch.cat([a, b], dim=2)
   return max_values , min_values

def train(train_loader, model, criterion, optimizer, conf,wmodel=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cpc = AverageMeter()
    scores = AverageAccMeter()
    end = time.time()
    model.train()

    cpc_loss_local = DeepLME()


    time_start = time.time()
    pbar = tqdm(train_loader, dynamic_ncols=True, total=len(train_loader))
    mixmethod = None
    clsw = None
    if 'mixmethod' in conf:
        if 'baseline' not in conf.mixmethod:
            mixmethod = conf.mixmethod
            if wmodel is None:
                wmodel = model

    for idx, (input, target) in enumerate(pbar):
        data_time.add(time.time() - end)
        input = input.cuda()
        target = target.cuda()
        maps = get_maps(input , target , model , conf)
        mask_p = maps.unfold(1, 32, 32).unfold(2, 32, 32)
        Pr = torch.sum(mask_p, dim=(3, 4))

        positives , negatives = get_triplets(Pr)
        P1 = torch.cat([positives , negatives] , dim=1)

        conf.mixmethod = 'snapmix'

        if 'baseline' not in conf.mixmethod:

            input,target_a,target_b,lam_a,lam_b = snapmix(input,target,conf,wmodel)

            output, a  , b , patches = model(input)

            cpc1 = cpc_loss_local(patches , P1)

            loss_a = criterion(output, target_a.cuda())
            loss_b = criterion(output, target_b.cuda())

            loss = 2 * torch.mean(loss_a * lam_a + loss_b * lam_b) + 0.01 * cpc1



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
