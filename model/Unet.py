import torch
from torch import nn
import json
import os


class Unet(nn.Module):
    def __init__(self, cfg: list, class_num: int, in_channel: int):
        super(Unet, self).__init__()
        # cfg = [64, 128, 256, 512, 1024]
        self.cfg = cfg

        self.conv1 = BasicBlock(in_channel, cfg[0])
        self.conv2 = BasicBlock(cfg[0], cfg[1])
        self.conv3 = BasicBlock(cfg[1], cfg[2])
        self.conv4 = BasicBlock(cfg[2], cfg[3])
        self.conv5 = BasicBlock(cfg[3], cfg[4])

        self.down_sample = down_sample()

        self.up_conv1 = up_conv(in_channel=cfg[4], out_channel=cfg[3])
        self.up_conv2 = up_conv(in_channel=cfg[3], out_channel=cfg[2])
        self.up_conv3 = up_conv(in_channel=cfg[2], out_channel=cfg[1])
        self.up_conv4 = up_conv(in_channel=cfg[1], out_channel=cfg[0])

        self.dropout = nn.Dropout(0.2)

        if class_num == 1 or class_num == 2:
            self.conv7 = nn.Conv2d(in_channels=cfg[0], out_channels=1, kernel_size=3, stride=1, padding=1)
            self.output = nn.Sigmoid()
        else:
            self.conv7 = nn.Conv2d(in_channels=cfg[0], out_channels=class_num, kernel_size=3, stride=1, padding=1)
            self.output = nn.Softmax(dim=1)

    def forward(self, x):
        # 下采样
        x1 = self.conv1(x)
        x2 = self.conv2(self.down_sample(x1))
        x3 = self.conv3(self.down_sample(x2))
        x4 = self.conv4(self.down_sample(x3))
        x5 = self.conv5(self.down_sample(x4))

        # 上采样
        x6 = self.dropout(self.up_conv1(x4, x5))
        x7 = self.dropout(self.up_conv2(x3, x6))
        x8 = self.dropout(self.up_conv3(x2, x7))
        x9 = self.dropout(self.up_conv4(x1, x8))

        x = self.conv7(x9)
        x = self.output(x)

        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# 下采样
class down_sample(nn.Module):
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super(down_sample, self).__init__()
        # print(kernel_size)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.maxpool(x)


# 上采样
class up_conv(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super(up_conv, self).__init__()
        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel//2),
            nn.ReLU(inplace=True)
        )
        self.conv = BasicBlock(in_channel=out_channel*2, out_channel=out_channel)

    def forward(self, x1, x2):
        x2 = self.up_sample(x2)
        x = torch.concat((x1, x2), dim=1)
        x = self.conv(x)
        return x


if __name__ == '__main__':
    cfg_path = './config.json'
    if not os.path.exists(cfg_path):
        cfg = [64, 128, 256, 512, 1024]
        with open(cfg_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'Unet': cfg}))
            cfg = json.load(f)
    else:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)

    model = Unet(cfg['Unet'], 2, 3)
    print(model)
