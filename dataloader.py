import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import glob

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class ChestXRayData(data.Dataset):
    def __init__(self, root, trans=None):
        self.root = root
        self.labels = {'NORMAL': 0, 
                       'PNEUMONIA': 1}
        self.imgs = glob.glob(os.path.join(self.root, '*/*'))
        
        if trans is None:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.trans = trans

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        img = self.trans(img)
        label = self.labels[self.imgs[index].split('/')[-2]]

        return img, label
