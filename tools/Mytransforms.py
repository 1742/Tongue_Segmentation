import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


# 为了能同时对图片和标签进行相同处理，覆写了一部分transforms的代码
class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target=None):
        image = F.resize(image, self.size)
        if target is not None:
            target = F.resize(target, self.size, interpolation=F.InterpolationMode.NEAREST)
            return image, target
        return image


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target=None):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                target = F.hflip(target)
                return image, target
        return image, target


class ToTensor(object):
    def __call__(self, image, target=None):
        image = F.to_tensor(image)
        if target is not None:
            if target.mode == 'P':
                target = torch.as_tensor(np.array(target), dtype=torch.int64)
            else:
                target = F.to_tensor(target)
            return image, target
        return image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        if target is not None:
            for t in self.transforms:
                image, target = t(image, target)
            return image, target
        else:
            for t in self.transforms:
                image = t(image)
            return image


class RGBToHSV(object):
    def __init__(self):
        self.smooth = 1e-5

    def __call__(self, image, target=None):
        image_hsv = self.rgb2hsv(image)
        if target is not None:
            target_hsv = self.rgb2hsv(target)
            return image_hsv, target_hsv
        return image_hsv

    def rgb2hsv(self, image):
        if image.mode != 'RGB':
            return image

        image = np.array(image, dtype=np.float32) / 255
        R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        Cmax = np.max(image, axis=2)
        Cmin = np.min(image, axis=2)
        d = Cmax - Cmin
        # 计算H
        H_R = np.array(R == Cmax, dtype=np.float32) * ((G - B) / (d + self.smooth))
        H_R_mask = np.array(H_R < 0, dtype=np.float32) * 360
        H_R = H_R + H_R_mask
        H_G = np.array(G == Cmax, dtype=np.float32) * ((B - R) / (d + self.smooth) + 120)
        H_B = np.array(B == Cmax, dtype=np.float32) * ((R - G) / (d + self.smooth) + 240)
        H = (H_R + H_G + H_B) / 2
        # 计算S
        S_0 = np.array(Cmax == 0, dtype=np.float32)
        S_1 = np.array(Cmax != 0, dtype=np.float32) * (d / (Cmax + self.smooth))
        S = (S_0 + S_1) * 255
        # 计算V
        V = Cmax * 255

        image_hsv = np.array([H, S, V], dtype='uint8').transpose(1, 2, 0)
        image_hsv = Image.fromarray(image_hsv)

        return image_hsv


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.colorjitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, target=None):
        image = self.colorjitter(image)
        if target is not None:
            if target.mode == 'RGB':
                target = self.colorjitter(target)
            return image, target
        return image


if __name__ == '__main__':
    test_img_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Tongue_Segmentation-master\data\train\image\43.png'

    img = Image.open(test_img_path).convert('RGB')
    t = RGBToHSV()
    y = t(img)

    plt.subplot(211)
    plt.imshow(y)

    plt.subplot(212)
    img_1 = cv2.imread(test_img_path)
    hsv = cv2.cvtColor(img_1, cv2.COLOR_BGR2HSV)
    plt.imshow(hsv)

    plt.show()


