import collections
import os
import pprint
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.io import imread
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.segmentation import slic, mark_boundaries
import cv2


def read_filepaths(file, mode):
    paths, labels = [], []
    with open(file, 'r') as f:
        lines = f.read().splitlines()

        for idx, line in enumerate(lines):
            if '/ c o' in line:
                break
            try:
                path, class_id , _ , _ , _ , _ = line.split(' ')
                path = '/media/data1/ameenali/exps/COVID-CT-DATASET/2A_images/' + path
                label = int(class_id)
            except:
                print(line)

            paths.append(path)
            labels.append(label)
    return paths, labels


def read_filepaths2(file):
    paths, labels = [], []
    with open(file, 'r') as f:
        lines = f.read().splitlines()

        for idx, line in enumerate(lines):
            if '/ c o' in line:
                break
            try:
                path , class_id , _ , _ , _ , _ = line.split(' ')
                path = '/media/data1/ameenali/exps/COVID-CT-DATASET/2A_images/' + path
                label = int(class_id)
            except:
                print(line)

            paths.append(path)
            labels.append(label)
    return paths, labels
def get_dataset(path):

 from os import listdir
 from os.path import isfile, join
 import os

 paths  = []
 labels = []

 for file in listdir(os.path.join(path , 'COVID19')):

  file_path = os.path.join(os.path.join(path , 'COVID19') , file)
  paths.append(file_path)
  labels.append(0)

 for file in listdir(os.path.join(path , 'NORMAL')):

  file_path = os.path.join(os.path.join(path , 'NORMAL') , file)
  paths.append(file_path)
  labels.append(1)

 for file in listdir(os.path.join(path , 'PNEUMONIA')):

  file_path = os.path.join(os.path.join(path , 'PNEUMONIA') , file)
  paths.append(file_path)
  labels.append(2)

 return paths , labels



class COVID_CT_Dataset(Dataset):
    """
    Code for reading the COVIDxDataset
    """

    def __init__(self, mode, n_classes=2, dataset_path='./datasets', dim=(224, 224)):
        self.mode = mode

        self.CLASSES = n_classes
        self.dim = dim
        self.COVIDxDICT = {'normal': 0, 'COVID-19': 1 , 'Penomia' : 2}
        trainfile = os.path.join('/media/data1/ameenali/exps/COVID-CT-DATASET' , 'train_COVIDx_CT-2A.txt')
        testfile = os.path.join('/media/data1/ameenali/exps/COVID-CT-DATASET'  , 'test_COVIDx_CT-2A.txt')

        #newtrainpath, newtrainlabel = read_filepaths2('../data/SARS-Cov-2/train_split.txt')
        #newtestpath, newtestlabel = read_filepaths2('../data/SARS-Cov-2/test_split.txt')

        if mode == 'train':
            train_folder = r'/media/data1/ameenali/Data/train'
            self.paths, self.labels = get_dataset(train_folder)
            #self.paths.extend(self.paths)
            #self.labels.extend(self.labels)
            #self.paths.extend(self.paths)
            #self.labels.extend(self.labels)

            #self.paths.extend(newtrainpath)
            #self.labels.extend(newtrainlabel)
#            c = list(zip(self.paths , self.labels))
#            random.shuffle(c)
#            self.paths , self.labels = zip(*c)

        elif mode == 'test':
            test_folder = r'/media/data1/ameenali/Data/test'
            self.paths, self.labels = get_dataset(test_folder)
            #self.paths.extend(newtestpath)
            #self.labels.extend(newtestlabel)

        print("{} examples =  {}".format(mode, len(self.paths)))
        self.mode = mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.mode == 'train':
         image_tensor , mask = self.load_image(self.paths[index])
         label_tensor = torch.tensor(self.labels[index], dtype=torch.long)
         return image_tensor, label_tensor
        image_tensor = self.load_image(self.paths[index])
        label_tensor = torch.tensor(self.labels[index], dtype=torch.long)
        return image_tensor, label_tensor


    def load_image(self, img_path):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))


        image = Image.open(img_path).convert('RGB')

        inputsize = 224
        transform = {
            'train': transforms.Compose(
                [transforms.Resize(256),
                 transforms.RandomCrop(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                 ]),
            'test': transforms.Compose(
                [transforms.Resize([inputsize, inputsize]),
                 ])
        }

        train_transformtotensor = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        test_transformtotensor = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        if self.mode == 'train':
            image = transform['train'](image)
        else:
            image = transform['test'](image)

        if self.mode == 'train':
         image_tensor = train_transformtotensor(image)
         return image_tensor,  img_path
        else:
         image_tensor = test_transformtotensor(image)

        return image_tensor
