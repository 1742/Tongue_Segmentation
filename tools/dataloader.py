import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tools.Mytransforms import Resize, RandomHorizontalFlip, RGBToHSV, ColorJitter, ToTensor, Compose
from PIL import Image
import numpy as np
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
        img = Image.open(os.path.join(self.data_path + '\\image', self.img_names[index])).convert('RGB')
        target = Image.open(os.path.join(self.data_path + '\\label', self.img_names[index])).convert('P')
        # 数据增强
        img, target = self.transformers(img, target)
        # 若target时灰度图或P图时，增加一个通道维度
        if len(target.size()) != 3:
            target = target.view(1, target.size(0), target.size(1))

        return img, target


if __name__ == '__main__':
    data_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Tongue_Segmentation-master\data\train'
    img_names = os.listdir(os.path.join(data_path, 'image'))

    transformers = [
        Resize(224),
        # RandomHorizontalFlip(),
        RGBToHSV(),
        ToTensor()
    ]

    test_datasets = MyDatasets(data_path, img_names, transformers)

    img1, target1 = test_datasets.__getitem__(1)
    img1 = img1.permute(1, 2, 0)
    target1 = target1.permute(1, 2, 0)
    # 将target转化成3通道
    target1_ = torch.concat((target1, target1, target1), dim=2)

    img2, target2 = test_datasets.__getitem__(2)
    img2 = img2.permute(1, 2, 0)
    target2 = target2.permute(1, 2, 0)
    # 将target转化成3通道
    target2_ = torch.concat((target2, target2, target2), dim=2)

    plt.subplot(2, 3, 1)
    plt.imshow(img1)
    plt.subplot(2, 3, 2)
    plt.imshow(target1)
    plt.subplot(2, 3, 3)
    plt.imshow(img1 * target1_)

    plt.subplot(2, 3, 4)
    plt.imshow(img2)
    plt.subplot(2, 3, 5)
    plt.imshow(target2)
    plt.subplot(2, 3, 6)
    plt.imshow(img2 * target2_)

    plt.show()



