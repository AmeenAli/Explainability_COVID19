import torch
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import h5py
import numpy as np
import os
from tqdm import tqdm
from utils.mixmethod import *

def predict(img, site, trans , model , n , loader , label):
    d = ['normal', 'COVID-19']
    input = trans(img)
#    groundtruth = torch.tensor(1) if label == 'COVID-19' else torch.tensor(0)

#    for idx , (train_img , train_label , site) in enumerate(loader):
#     if idx == n:
#      break
#     print(train_label)
#     exit(-1)
#     input = input.repeat(32 , 1 , 1 , 1)
#     groundtruth = groundtruth.repeat(32)
#     scores , features , _ , _ = model(input)
#     print(F.softmax(scores , dim=1))
#     exit(-1)

    img = input.unsqueeze(0)
    input = Variable(img.cuda())
    score, features , _ , _ = model(input)
    probability = torch.nn.functional.softmax(score, dim=1)
    max_value, index = torch.max(probability, 1)

    return d[index.item()], probability


def test(model , loader , n=3):
 model.eval()
 dataset = ['../data/COVID-CT',
           '../data/SARS-Cov-2'
           ]
 dict = {'normal': 0, 'COVID-19': 1}
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
     score = torch.from_numpy(np.array([]))
     target = torch.from_numpy(np.array([]))
     for input in tqdm(testimages):
         # print(input)
         label = input.split(' ')[-1][:-1]
         #print(label)
         #exit(-1)
         if dataset_path != '../data/COVID-CT':
             image = input.split(' ')[1] + ' ' + input.split(' ')[2]
             site = 'new'
             trans = transforms.Compose(
                 [transforms.Resize([224, 224]),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
         else:
             image = '../data/COVID-CT/test/' + input.split(' ')[1]
             site = 'ucsd'
             trans = transforms.Compose(
                 [transforms.Resize([224, 224]),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
         img = Image.open(image).convert('RGB')




         pred, prob = predict(img, site, trans , model , n , loader , label)
         if label == pred:
             correct += 1
         if label == 'COVID-19':
             covid_all += 1
         if pred == 'COVID-19':
             covid_predict += 1
         if label == 'COVID-19' and pred == 'COVID-19':
             covid_correct += 1
        # print(input.split(' ')[1], label, pred, '\t\t', correct, '/', len(testimages))
     accuracy = float(correct) / float(len(testimages))
     recall = float(covid_correct) / float(covid_all)
     precision = float(covid_correct) / float(covid_predict)
     f1 = 2 * recall * precision / (recall + precision)

     print('{}\tCorrect: {:d}/{:d}, Accuracy: {:.2f}, Recall: {:.2f}, Precision: {:.2f}, F1: {:.2f}'.format(
         dataset_path[8:],
         correct,
         len(testimages),
         accuracy * 100.,
         recall * 100.,
         precision * 100.,
         f1 * 100.
     ))
