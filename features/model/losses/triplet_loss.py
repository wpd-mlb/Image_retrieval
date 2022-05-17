'''
@Author      : WangPeide
@Contact     : 377079627@qq.com
LastEditors: Please set LastEditors
@Description :
LastEditTime: 2022/1/25 23:45
'''
import torch.nn as nn
import torch


# reference
# https://blog.csdn.net/qq_32523711/article/details/103826600
class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """
    def __init__(self,margin=0.6,weight=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

        self.ranking_loss = nn.MarginRankingLoss(margin=margin)  # 获得一个简单的距离triplet函数
        # if y = 1y=1 then it assumed the first input should be ranked higher (have a larger value) than the second input,
        # and vice-versa for y = -1y=−1 .
        self.weight = weight
    def forward(self, inputs, labels):
        n = inputs.size(0)  # 获取batch_size
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)  # 每个数平方后， 进行加和（通过keepdim保持2维），再扩展成nxn维
        dist = dist + dist.t()  # 这样每个dis[i][j]代表的是第i个特征与第j个特征的平方的和
        dist.addmm_(beta=1, alpha=-2, mat1=inputs, mat2=inputs.t())  # 然后减去2倍的 第i个特征*第j个特征 从而通过完全平方式得到 (a-b)^2
        dist = dist.clamp(min=1e-12).sqrt()  # 然后开方

        mask = labels.expand(n, n).eq(labels.expand(n, n).t())  # 这里dist[i][j] = 1代表i和j的label相同， =0代表i和j的label不相同
        dist_ap, dist_an = [], []
        for i in range(n):
            if len(dist[i][mask[i]])>0 and len(dist[i][mask[i] == 0])>0:
                dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # 在i与所有有相同label的j的距离中找一个最大的
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # 在i与所有不同label的j的距离找一个最小的
        dist_ap = torch.cat(dist_ap)  # 将list里的tensor拼接成新的tensor
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)  # 声明一个与dist_an相同shape的全1tensor

        loss = self.ranking_loss(dist_an, dist_ap, y)

        if self.training:
            return loss*self.weight
        else:
            return None

if __name__ =="__main__":
    pass
    # triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    # anchor = torch.randn(20, 128, requires_grad=True)
    # positive = torch.randn(20, 128, requires_grad=True)
    # negative = torch.randn(20, 128, requires_grad=True)
    # output = triplet_loss(anchor, positive, negative)
    # output.backward()
    # print(output)

    inputs = torch.arange(1, 13).view(3, 4).float()
    n = inputs.size(0)  # 获取batch_size
    dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)  # 每个数平方后， 进行加和（通过keepdim保持2维），再扩展成nxn维
    dist = dist + dist.t()  # 这样每个dis[i][j]代表的是第i个特征与第j个特征的平方的和
    dist.addmm_(beta=1, alpha=-2, mat1=inputs, mat2=inputs.t())  # 然后减去2倍的 第i个特征*第j个特征 从而通过完全平方式得到 (a-b)^2
    dist = dist.clamp(min=1e-12).sqrt()  # 然后开方
    print(dist)
    # targets = torch.arange(1,4)
    targets = torch.tensor([1, 1, 2])
    mask = targets.expand(n, n).eq(targets.expand(n, n).t())
    print(mask)
    dist_ap, dist_an = [], []
    for i in range(n):
        if len(dist[i][mask[i]]) > 0 and len(dist[i][mask[i] == 0]) > 0:
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # 在i与所有有相同label的j的距离中找一个最大的
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # 在i与所有不同label的j的距离找一个最小的
    print(dist_ap)
    dist_ap = torch.cat(dist_ap)  # 将list里的tensor拼接成新的tensor
    dist_an = torch.cat(dist_an)
    print(dist_ap)









