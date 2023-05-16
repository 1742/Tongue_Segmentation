import numpy as np
import torch
import matplotlib.pyplot as plt
import json


def Accuracy(predict: torch.Tensor, target: torch.Tensor, is_batch: bool):
    if is_batch:
        num = predict.size(2) * predict.size(3)
        acc = 0
        for bs in range(predict.size(0)):
            for cls in range(predict.size(1)):
                per_cls_acc = 0
                # 各类别准确率
                per_cls_acc += torch.sum(predict[bs, cls] == target[bs, cls]) / num
            # 单张图片所有类别准确率之和
            acc += per_cls_acc
        # 所有类别准确率平均
        acc /= predict.size(0)
    else:
        num = predict.size(2) * predict.size(3)
        acc = 0
        for cls in range(predict.size(0)):
            per_cls_acc = 0
            # 各类别准确率
            per_cls_acc += torch.sum(predict[cls] == target[cls]) / num
        # 单张图片所有类别准确率之和
        acc += per_cls_acc

    return float(acc)


def mIOU(predict: torch.Tensor, target: torch.Tensor, is_batch: bool):
    # 标签采用独热编码
    predict = predict.long().cpu()
    target = target.long().cpu()

    # 初始化mIou
    mIou = 0
    if is_batch:
        # 分批次计算每一张图的IOU
        for bs in range(predict.size(0)):
            per_cls_Iou = 0
            # 0代表背景，从1开始
            for cls in range(predict.size(1)):
                # 交集
                per_intersection = np.logical_and(predict[bs, cls] == 1, target[bs, cls] == 1)
                # 并集
                union = np.logical_or(predict[bs, cls] == 1, target[bs, cls] == 1)
                # 单张图片各类别的IOU
                per_cls_Iou += torch.sum(per_intersection) / torch.sum(union)
            # 单张图片的mIOU
            mIou += per_cls_Iou / predict.size(1)
        # 取批次mIOU平均
        mIou /= predict.size(0)

    else:
        per_cls_Iou = 0
        # 0代表背景，从1开始
        for cls in range(predict.size(0)):
            # 交集
            per_intersection = np.logical_and(predict[cls] == 1, target[cls] == 1)
            # 并集
            union = np.logical_or(predict[cls] == 1, target[cls] == 1)
            # 单张图片各类别的IOU
            per_cls_Iou += torch.sum(per_intersection) / torch.sum(union)
        # 单张图片的mIOU
        mIou += per_cls_Iou / predict.size(0)

    return float(mIou)


def Visualization(evaluation, train: bool, save_option: str = None):
    index = list(evaluation.keys())
    if train:
        index.remove('epoch')
        epoch = range(1, evaluation['epoch'] + 1)
        for i, k in enumerate(index):
            plt.plot(epoch, evaluation[k][0], label='train')
            plt.plot(epoch, evaluation[k][1], label='val')
            plt.title('train' + k)
            plt.xlabel('epoch')
            plt.ylabel(k)
            plt.legend()
            plt.grid()
            plt.show()

    else:
        index.remove('num')
        num = range(1, evaluation['num'] + 1)
        for i, k in enumerate(index):
            plt.plot(num, evaluation[k], label='test')
            plt.title('test' + k)
            plt.xlabel('num')
            plt.ylabel(k)
            plt.legend()
            plt.grid()
            plt.show()


if __name__ == '__main__':
    # predict = torch.rand((16, 3, 224, 224)) > 0.5
    # target = torch.rand((16, 3, 224, 224)) > 0.5
    # print('acc:', Accuracy(predict, target, True))
    # print('mIOU:', mIOU(predict, target, True))

    effect_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Tongue_Segmentation-master\runs\effect_1\effect.json'

    with open(effect_path, 'r', encoding='utf-8') as f:
        effect = json.load(f)

    Visualization(effect, True)
