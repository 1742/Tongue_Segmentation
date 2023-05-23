import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from tools.Mytransforms import Resize, RandomHorizontalFlip, ToTensor, Compose
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys


class MyDatasets(Dataset):
    def __init__(self, data_path: str, img_names: list, transformers: [list, dict]):
        super(MyDatasets, self).__init__()
        self.data_path = data_path
        self.img_names = img_names
        self.transformers = Compose(transformers)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name, label = self.img_names[index].split(' ')
        img_path = os.path.join(self.data_path, label)
        img = Image.open(os.path.join(img_path + '/image', img_name)).convert('RGB')
        target = Image.open(os.path.join(img_path + '/tongue', img_name)).convert('L')
        # 数据增强
        img, target = self.transformers(img, target)
        # 若target时灰度图或P图时，增加一个通道维度
        if len(target.size()) != 3:
            target = target.view(1, target.size(0), target.size(1))

        return img, target


def shuffle(data: list):
    np.random.shuffle(data)
    return data
