import torch
from torch import nn


# 自定义损失函数
class Dice(nn.Module):
    def __init__(self, device: str, smooth: float = 1e-17):
        super(Dice, self).__init__()
        # diceloss只是简单数字加减乘除计算，nn.Module会自动计算此类损失的梯度，不需自定义backward？？？
        self.smooth = smooth
        self.device = device

    def forward(self, predict: torch.Tensor, target: torch.Tensor):
        if self.device == 'cpu':
            predict = predict.long().cpu()
            target = target.long().cpu()
        else:
            predict = predict.long().cuda()
            target = target.long().cuda()

        dice_loss = 0
        for bs in range(predict.size(0)):
            intersection = torch.sum(predict[bs] * target[bs])
            union = torch.sum(torch.sum(predict[bs])) + torch.sum(target[bs])
            dice_loss += 1 - (2 * intersection) / union
        # 返回批次平均diceloss
        return dice_loss / predict.size(0)


class BCE_and_Dice_Loss(nn.Module):
    def __init__(self, device: str, beta: float = 0.5):
        super(BCE_and_Dice_Loss, self).__init__()
        self.device = device
        self.beta = beta
        self.bce = nn.BCELoss()
        self.dice = Dice(device=device)

    def forward(self, predict: torch.Tensor, target: torch.Tensor):
        loss = self.beta * self.bce(predict, target) + (1 - self.beta) * self.dice(predict, target)

        return loss


if __name__ == '__main__':
    predict = torch.rand((16, 1, 224, 224))
    target = torch.rand((16, 1, 224, 224)) > 0.5

