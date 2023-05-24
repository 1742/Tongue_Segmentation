import sys

import torch
from torch import nn
from model.Unet import *
from tools.my_loss import Dice, BCE_and_Dice_Loss

from torch.utils.data import Dataset, DataLoader
from tools.dataloader import MyDatasets, shuffle
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tools import Mytransforms
import numpy as np

import os
import json
import random
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools.evaluation_index import Accuracy, mIOU, Visualization


data_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Tongue_Segmentation-master\data'
img_names_txt = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Tongue_Segmentation-master\data\img_names.txt'
cfg_file = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Tongue_Segmentation-master\model\config.json'
indicator_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Tongue_Segmentation-master\runs'
pretrained_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Tongue_Segmentation-master\model\Unet.pth'
effect_path = r'runs\result\effect.json'
save_figure_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Tongue_Segmentation-master\runs\result.png'
save_picture_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Tongue_Segmentation-master\runs\result'


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('The predict will run in {} ...'.format(device))
pretrained = True
save_option = True


def predict(
    device: str,
    model: nn.Module,
    pretrained_path: str,
    test_datasets: Dataset,
    criterion_name: str,
    save_option: bool
):

    test_loss = []
    test_acc = []
    test_mIou = []

    # 加载权重
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, torch.device(device)))
        print('Successfully load pretrained model from {}'.format(pretrained_path))
    else:
        print('model parameters files is not exist!')
        sys.exit(0)
    model.to(device)

    test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=False)

    if criterion_name == 'BCELoss':
        criterion = nn.BCELoss()
    elif criterion_name == 'BCE_and_Dice_Loss':
        criterion = BCE_and_Dice_Loss(device=device, beta=0.6)

    model.eval()
    with tqdm(total=len(test_dataloader)) as pbar:
        pbar.set_description('loading: ')

        with torch.no_grad():
            for i, (img, target) in enumerate(test_dataloader):
                img = img.to(device, dtype=torch.float)
                target = target.to(device, dtype=torch.float)

                output = model(img).float()
                pred = output > 0.5

                loss = criterion(output, target)
                acc = Accuracy(pred, target, False)
                mIou = mIOU(pred, target, False)

                # 记录指标
                test_loss.append(loss.item())
                test_acc.append(acc)
                test_mIou.append(mIou)

                pbar.set_postfix({criterion_name: loss.item(), 'acc': acc, 'mIOU': mIou})

                pbar.update(1)

                if save_option:
                    img = np.array((img * 255).squeeze().permute(1, 2, 0))
                    # img_hsv = np.array((img * 255).squeeze().permute(1, 2, 0))
                    # img = Image.open(*img_path).convert('RGB')
                    # 将P格式的target转化成3通道RGB格式
                    target = target * 255
                    target = np.array(torch.cat((target, target, target), dim=1).squeeze().permute(1, 2, 0))
                    # 分割舌体
                    pred = pred.long()
                    pred = np.array(torch.cat((pred, pred, pred), dim=1).squeeze().permute(1, 2, 0))
                    pred = img * pred

                    # 记录图片大小
                    size = img.shape
                    # size = img_hsv.shape

                    # 转化为Image格式
                    img = Image.fromarray(np.uint8(img))
                    # img_hsv = Image.fromarray(np.uint8(img_hsv))
                    target = Image.fromarray(np.uint8(target))
                    pred = Image.fromarray(np.uint8(pred))

                    # 拼接三张图片
                    result = Image.new('RGB', (size[0] * 3, size[1]), (0, 0, 0))
                    result.paste(img, (0, 0))
                    # result.paste(img_hsv, (size[0] + 1, 0))
                    result.paste(target, (size[0] + 1, 0))
                    result.paste(pred, (2 * size[0] + 1, 0))

                    plt.imshow(result)
                    plt.title('loss: {:.3f}  acc: {:.3f}  IOU: {:.3f}'.format(loss, acc, mIou))
                    plt.axis('off')
                    plt.savefig(save_picture_path + '\\{}.png'.format(i))

    return {'num': len(test_dataloader), 'loss': test_loss, 'acc': test_acc, 'mIOU': test_mIou}


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
        # Mytransforms.RGBToHSV(),
        Mytransforms.ToTensor()
    ]

    test_datasets = MyDatasets(data_path, test_data, transformers)

    # 读取模型结构
    with open(cfg_file, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    model = Unet(cfg['Unet'], 1, 3)

    criterion = 'BCE_and_Dice_Loss'
    print('loss:', criterion)

    effect = predict(
        device=device,
        model=model,
        test_datasets=test_datasets,
        criterion_name=criterion,
        pretrained_path=pretrained_path,
        save_option=save_option
    )

    with open(effect_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(effect))

    # Visualization(effect, False)
