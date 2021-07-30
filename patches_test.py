import torch
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import h5py
import numpy as np
import os
from tqdm import tqdm
from utils.mixmethod import *
from sklearn import linear_model
from matplotlib import pyplot as plt
import cv2


def save(cams, zero_score_list, name, mode='plt'):
    if mode == 'csv':
        cpath = os.path.join('excels', name + '.csv')
        f = open(cpath, 'w')
        f.write('SN,cams,cams_train,zero_score_list,one_score_list\n')
        f.flush()
        for i in range(len(cams)):
            f.write(str(i) + ',' +
                    str(cams[i]) + ',' +
                    str(zero_score_list[i]) + ',' +
                    '\n')
            f.flush()
    else:
        a = os.listdir(os.path.join('excels', name))
        full_name = str(len(a)) + '_' + name
        cpath = os.path.join('excels', name, full_name + '.png')
        plt.plot(cams, zero_score_list, marker='o')
        plt.grid()
        plt.savefig(cpath)
        plt.close()


def save_imgs(imgs, zero_score_list):
    for inx, img in enumerate(imgs):
        score = 1 - zero_score_list[inx]
        a = os.listdir('excels/imgs')
        cpath = os.path.join('excels', 'imgs', str(len(a)) + '_' + str(score) + '.png')
        img = img.permute(1, 2, 0)
        img = img.detach().cpu().numpy().copy()
        img = (img - img.min()) / (img.max() - img.min())
        cv2.imwrite(cpath, 255 * img)


def get_bbox(inx, i, pSize):
    x, y = inx[i]
    radius = int(pSize/2)
    x = int((x.item() + 0.5) * pSize)
    y = int((y.item() + 0.5) * pSize)
    return x - radius, y - radius, x + radius, y + radius


def get_name(pred, gt):
    if pred == 1:
        if gt == 1:
            return 'hit'
        else:
            return 'fa'
    else:
        if gt == 1:
            return 'md'
        else:
            return 'rej'


def decision(zero_score):
    if zero_score > 0.5:
        return torch.tensor(0)
    else:
        return torch.tensor(1)


def get_inx(mask, k=100):
    index = torch.topk(mask.reshape(-1), k, dim=0)[1][..., None]
    a = index // mask.shape[0]
    b = index % mask.shape[0]
    max_values = torch.cat([a, b], dim=1)
    return max_values


def is_flip(new, old, th=0.5):
    if new > 1 - th and old == 1:
        return 1
    elif new <= th and old == 0:
        return 1
    else:
        return 0


def remove_random_boxes(img, target, model, k, pSize, th):
    score, _, _, _ = model(img)
    zero_score = F.softmax(score[0], dim=0)[0].item()
    pred = decision(zero_score)
    maps, _ = get_corona_maps(img, pred.repeat(2), model)
    patches = maps.unfold(0, pSize, pSize).unfold(1, pSize, pSize)
    patches = torch.sum(patches, dim=(2, 3))
    inx = get_inx(patches, k)
    cams = []
    total_sum = 0
    zero_score_list = []
    flip_list = []
    imgs_list = []
    cams.append(0)
    flip_list.append(0)
    # imgs_list.append(img[0].clone())
    # zero_value = img[0].min()
    zero_score_list.append(zero_score)
    for i in range(inx.shape[0]):
        bbx1, bby1, bbx2, bby2 = get_bbox(inx, i, pSize)
        current_sum = maps[bbx1:bbx2, bby1:bby2].sum()
        cams.append(total_sum + current_sum.item())
        img[0, :, bbx1:bbx2, bby1:bby2] = torch.zeros([1, 3, bbx2 - bbx1, bby2 - bby1]).float().cuda()
        new_scores = model(img)[0]
        new_scores = F.softmax(new_scores, dim=1)
        zero_score_list.append(new_scores[0, 0].item())
        total_sum = cams[-1]
        flip_stat = is_flip(new_scores[0, 0].item(), pred.item(), th=th)
        flip_list.append(flip_stat)
        imgs_list.append(img[0].clone())
    # if np.mean(flip_list) > 0:
    #     name = get_name(pred.item(), target[0].item())
    #     save(cams, flip_list, name)
    if np.mean(flip_list) > 0.5:
        if pred:
            # save_imgs(imgs_list, zero_score_list)
            return torch.tensor([1, 0])
        else:
            # save_imgs(imgs_list, zero_score_list)
            return torch.tensor([0, 1])
    return torch.tensor([zero_score, 1-zero_score])


def save_maps(maps):
    a = os.listdir('excels/maps')
    cpath = os.path.join('excels', 'maps', str(len(a)) + '.png')
    map = maps.detach().cpu().numpy().copy()
    map = (map - map.min()) / (map.max() - map.min())
    cv2.imwrite(cpath, 255*map)


def predict(img, trans, model, n, label, pSize, th):
    d = ['normal', 'COVID-19']
    input = Variable(trans(img).unsqueeze(0).cuda())
    input = input.repeat(2, 1, 1, 1)
    target = torch.tensor(1) if label == 'COVID-19' else torch.tensor(0)
    new_score = remove_random_boxes(input, target.repeat(2), model, n, pSize, th)
    probability, index = torch.max(new_score, dim=0)
    return d[index], probability


def test(model, loader, k=100, pSize=8, th=0.3):
    model.eval()
    dataset = ['../data/COVID-CT', '../data/SARS-Cov-2']
    dict = {'normal': 0, 'COVID-19': 1}
    info_list = []
    for dataset_path in dataset:
        print('Start testing', dataset_path[8:])
        correct = 0
        covid_correct = 0
        covid_predict = 0
        covid_all = 0
        files = os.listdir(dataset_path)
        images = []
        dirss = os.listdir(dataset_path)
        dirs = []
        for a in dirss:
            if 'txt' in a:
                continue
            if 'zip' in a:
                continue
            if 'val' in a:
                continue
            if 'train' in a:
                continue
            if 'test' in a:
                continue
            dirs.append(a)
        testfile = dataset_path + '/test_split.txt'
        testimages = open(testfile, 'r').readlines()
        for input in tqdm(testimages):
            label = input.split(' ')[-1][:-1]
            if dataset_path != '../data/COVID-CT':
                image = input.split(' ')[1] + ' ' + input.split(' ')[2]
                site = 'new'
                trans = transforms.Compose(
                                         [transforms.Resize([224, 224]),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                )
            else:
                image = '../data/COVID-CT/test/' + input.split(' ')[1]
                site = 'ucsd'
                trans = transforms.Compose(
                    [transforms.Resize([224, 224]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                )
            img = Image.open(image).convert('RGB')
            pred, prob = predict(img, trans, model, k, label, pSize, th)
            if label == pred:
                correct += 1
            if label == 'COVID-19':
                covid_all += 1
            if pred == 'COVID-19':
                covid_predict += 1
            if label == 'COVID-19' and pred == 'COVID-19':
                covid_correct += 1
        accuracy = float(correct) / (float(len(testimages)) + 1e-3)
        recall = float(covid_correct) / (float(covid_all) + 1e-3)
        precision = float(covid_correct) / (float(covid_predict) + 1e-3)
        f1 = 2 * recall * precision / (recall + precision + 1e-3)

        info = '{}\tCorrect: {:d}/{:d}, Accuracy: {:.2f}, Recall: {:.2f}, Precision: {:.2f}, F1: {:.2f}'.format(
            dataset_path[8:],
            correct,
            len(testimages),
            accuracy * 100.,
            recall * 100.,
            precision * 100.,
            f1 * 100.)
        info_list.append(info)
    return info_list


def get_corona_maps(input, target, model):
    imgsize = (224, 224)
    bs = input.size(0)
    with torch.no_grad():
        output, fms, _, _ = model(input)
        clsw = model.classifier
        weight = clsw.weight.data
        bias = clsw.bias.data
        weight = weight.view(weight.size(0), weight.size(1), 1, 1)
        fms = F.relu(fms)
        poolfea = F.adaptive_avg_pool2d(fms,(1,1)).squeeze()
        clslogit = F.softmax(clsw.forward(poolfea))
        logitlist = []
        for i in range(bs):
            logitlist.append(clslogit[i, target[i]])
        clslogit = torch.stack(logitlist)
        out = F.conv2d(fms, weight, bias=bias)
        outmaps = []
        for i in range(bs):
            evimap = out[i,target[i]]
            outmaps.append(evimap)
        outmaps = torch.stack(outmaps)
        if imgsize is not None:
            outmaps = outmaps.view(outmaps.size(0), 1, outmaps.size(1), outmaps.size(2))
            outmaps = F.interpolate(outmaps, imgsize, mode='bilinear', align_corners=False)
        outmaps = outmaps.squeeze()
        for i in range(bs):
            outmaps[i] -= outmaps[i].min()
            outmaps[i] /= outmaps[i].sum()
    return outmaps[0], clslogit


