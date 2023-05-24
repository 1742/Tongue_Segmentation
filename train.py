import sys

import torch
from torch import nn
from model.Unet import *
from tools.my_loss import Dice, BCE_and_Dice_Loss

from torch.utils.data import Dataset, DataLoader
from tools.dataloader import MyDatasets, shuffle
from torchvision import transforms
from tools import Mytransforms
import numpy as np

import os
import json
import random
from tqdm import tqdm
from tools.evaluation_index import Accuracy, mIOU, Visualization


data_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Tongue_Segmentation-master\data'
img_names_txt = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Tongue_Segmentation-master\data\img_names.txt'
cfg_file = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Tongue_Segmentation-master\model\config.json'
indicator_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Tongue_Segmentation-master\runs'
pretrained_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Tongue_Segmentation-master\model\Unet.pth'
save_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Tongue_Segmentation-master\model'
effect_path = r'runs/effect/effect.json'
save_figure_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Tongue_Segmentation-master\runs\result\result.png'

learning_rate = 1e-4
weight_decay = 1e-8
epochs = 5
batch_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('The train will run in {} ...'.format(device))
pretrained = False
save_option = True


def train(
        device: str,
        model: nn.Module,
        train_datasets: Dataset,
        val_datasets: Dataset,
        batch_size: int,
        epochs: int,
        lr: float,
        weight_decay: float,
        optim: str,
        criterion_name: str,
        pretrained: bool,
        save_option: bool,
        lr_schedule: dict = None
        ):

    # 记录验证集最好状态，并提前结束训练
    # best_val = 0
    # flag = 0

    # 返回指标
    train_loss = []
    train_acc = []
    train_mIou = []
    val_loss = []
    val_acc = []
    val_mIou = []

    if pretrained:
        if os.path.exists(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path, torch.device(device)))
            print('Successfully load pretrained model from {}'.format(pretrained_path))
        else:
            print('model parameters files is not exist!')
            sys.exit(0)
    model.to(device)

    if batch_size == 1:
        is_batch = False
    else:
        is_batch = True

    train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_datasets, batch_size=batch_size, shuffle=True)

    if optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if criterion_name == 'BCELoss':
        criterion = nn.BCELoss()
    elif criterion_name == 'BCE_and_Dice_Loss':
        criterion = BCE_and_Dice_Loss(device=device, beta=0.6)

    if lr_schedule is None:
        lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    elif lr_schedule['name'] == 'StepLR':
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_schedule['step_size'], gamma=lr_schedule['gamma'])
    elif lr_schedule['name'] == 'ExponentialLR':
        lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_schedule['gamma'])

    for epoch in range(epochs):
        # 训练

        # model.train()的作用是启用 Batch Normalization 和 Dropout
        model.train()

        per_train_loss = 0
        per_train_acc = 0
        per_train_mIou = 0
        with tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description('epoch - {} train'.format(epoch+1))

            for i, (img, target) in enumerate(train_dataloader):
                img = img.to(device, dtype=torch.float)
                target = target.to(device, dtype=torch.float)

                output = model(img).float()

                loss = criterion(output, target)
                acc = Accuracy(output > 0.5, target, is_batch)
                mIou = mIOU(output > 0.5, target, is_batch)

                # 记录每批次平均指标
                per_train_loss += loss.item()
                per_train_acc += acc
                per_train_mIou += mIou

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_schedule.step()

                pbar.set_postfix({criterion_name: loss.item(), 'acc': acc, 'mIOU': mIou})

                pbar.update(1)

        # 记录每训练次平均指标
        train_loss.append(per_train_loss / len(train_dataloader))
        train_acc.append(per_train_acc / len(train_dataloader))
        train_mIou.append(per_train_mIou / len(train_dataloader))

        # 验证
        # model.eval()的作用是禁用 Batch Normalization 和 Dropout
        model.eval()

        per_val_loss = 0
        per_val_acc = 0
        per_val_mIou = 0

        with tqdm(total=len(val_dataloader)) as pbar:
            pbar.set_description('epoch - {} val'.format(epoch + 1))

            with torch.no_grad():
                for i, (img, target) in enumerate(val_dataloader):

                    img = img.to(device, dtype=torch.float)
                    target = target.to(device, dtype=torch.float)

                    output = model(img).float()

                    loss = criterion(output, target)
                    acc = Accuracy(output > 0.5, target, is_batch)
                    mIou = mIOU(output > 0.5, target, is_batch)

                    # 记录指标
                    per_val_loss += loss.item()
                    per_val_acc += acc
                    per_val_mIou += mIou

                    pbar.set_postfix({criterion_name: loss.item(), 'acc': acc, 'mIOU': mIou})

                    pbar.update(1)

        # 记录每训练次平均指标
        val_loss.append(per_val_loss / len(val_dataloader))
        val_acc.append(per_val_acc / len(val_dataloader))
        val_mIou.append(per_val_mIou / len(val_dataloader))

        # record_val_loss = per_val_loss / len(val_dataloader)
        # record_val_acc = per_val_acc / len(val_dataloader)
        # record_val_mIOU = per_val_mIou / len(val_dataloader)

        # 提前结束条件
        # if (record_val_acc + record_val_mIOU - record_val_loss) > best_val:
        #     best_val = per_val_acc + per_val_mIou - per_val_loss
        # if (record_val_acc + record_val_mIOU - record_val_loss) < best_val and record_val_acc > 0.9 and record_val_mIOU > 0.8:
        #     flag = 1
        #
        # if flag:
        #     break

    if save_option:
        torch.save(model.state_dict(), os.path.join(save_path, 'Unet.pth'))

    return {'epoch': epoch+1, 'loss': [train_loss, val_loss], 'acc': [train_acc, val_acc],'mIOU': [train_mIou, val_mIou]}


if __name__ == '__main__':
    # 读取图片名
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
    print('Successfully read image names from {}'.format(img_names_txt))
    img_names = shuffle(img_names)

    # 划分数据集
    data_num = len(img_names)
    train_data = img_names[:int(data_num * 0.7)]
    val_data = img_names[int(data_num * 0.7):int(data_num * 0.9)]
    test_data = img_names[int(data_num * 0.9):]
    print('train_data_num:', len(train_data))
    print('val_data_num:', len(val_data))
    print('test_data_num:', len(test_data))

    transformers = [
        Mytransforms.Resize((224, 224)),
        # Mytransforms.RandomHorizontalFlip(),
        # Mytransforms.RGBToHSV(),
        Mytransforms.ToTensor()
    ]

    train_datasets = MyDatasets(data_path, train_data, transformers)
    val_datasets = MyDatasets(data_path, val_data, transformers)

    # 读取模型结构
    with open(cfg_file, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    model = Unet(cfg['Unet'], 1, 3)
    optimizer = 'Adam'
    # criterion = 'BCELoss'
    criterion = 'BCE_and_Dice_Loss'
    # lr_schedule = {'name': 'ExponentialLR', 'gamma': 0.99}
    lr_schedule = None
    print('loss:', criterion)
    print('optimizer:', optimizer)
    print('lr_schedule:', lr_schedule)

    effect = train(
        device=device,
        model=model,
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        batch_size=batch_size,
        epochs=epochs,
        lr=learning_rate,
        weight_decay=weight_decay,
        optim=optimizer,
        criterion_name=criterion,
        pretrained=pretrained,
        save_option=save_option,
        lr_schedule=lr_schedule
    )

    with open(effect_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(effect))

    Visualization(effect, True)

