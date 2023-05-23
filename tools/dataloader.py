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
        img_name, label = self.img_names[index].split(' ')
        img_path = os.path.join(self.data_path, label)
        img = Image.open(os.path.join(img_path + '\\image', img_name)).convert('RGB')
        target = Image.open(os.path.join(img_path + '\\tongue', img_name)).convert('L')
        # 数据增强
        img, target = self.transformers(img, target)
        # 若target时灰度图或P图时，增加一个通道维度
        if len(target.size()) != 3:
            target = target.view(1, target.size(0), target.size(1))

        return img, target


def shuffle(data: list):
    np.random.shuffle(data)
    return data


if __name__ == '__main__':
    data_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Tongue_Segmentation-master\data'
    img_names_txt = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Tongue_Segmentation-master\data\img_names.txt'

    if not os.path.exists(img_names_txt):
        img_names = []
        for sx_img_name in os.listdir(os.path.join(data_path, 'sx\\image')):
            img_names.append(sx_img_name + ' sx')
        for xx_img_name in os.listdir(os.path.join(data_path, 'xx\\image')):
            img_names.append(xx_img_name + ' xx')

        with open(img_names_txt, 'w', encoding='utf-8') as f:
            for img_name in img_names:
                f.write(img_name)
                f.write('\n')

        print('Successfully generated img_names.txt in {}'.format(img_names_txt))

    img_names = []
    with open(img_names_txt, 'r', encoding='utf-8') as f:
        for img_name in f.readlines():
            img_names.append(img_name.strip())

    transformers = [
        Resize((224, 224)),
        # RandomHorizontalFlip(),
        # RGBToHSV(),
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



