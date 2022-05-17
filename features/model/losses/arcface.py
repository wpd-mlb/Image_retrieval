'''
@Author      : WangPeide
@Contact     : 377079627@qq.com
LastEditors: Please set LastEditors
@Description :
LastEditTime: 2022/1/17 23:41
'''

import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import math
import numpy as np

# paper with code implement
# https://github.com/ronghuaiyang/arcface-pytorch/blob/47ace80b128042cd8d2efd408f55c5a3e156b032/models/metrics.py#L10
# https://github.com/HuangYG123/CurricularFace/blob/master/head/metrics.py
# CurricularFaceLoss
class ArcfaceLoss_Dropout(nn.Module):
    r"""Implement of large margin arc distance: :
         Args:
             in_feat: size of each input sample
             num_classes: size of each output sample
             scale: norm of input feature
             margin: margin   init_0.5
             cos(theta + m)   init_64
         """
    def __init__(self, in_feat, num_classes,scale=64,margin=0.35,dropout_rate=0.3,weight=1.0):
        super(ArcfaceLoss_Dropout,self).__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = scale
        self._m = margin
        self.weight_loss = weight

        self.weight = nn.Parameter(torch.Tensor(num_classes, in_feat))
        self.register_buffer('t', torch.zeros(1))  # buffer参数
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(1))
        self.cos_m = math.cos(self._m)
        self.sin_m = math.sin(self._m)
        self.threshold = math.cos(math.pi - self._m)
        self.mm = math.sin(math.pi - self._m) * self._m # 阈值，避免theta + m >= pi

        self.criterion = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, features, targets):
        # backbone —>[n,v]
        # weight [v,c]->v是特征维度，c是类别数
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(input=self.dropout(F.normalize(features)), weight=F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

        target_logit = cos_theta[torch.arange(0, features.size(0)), targets].view(-1, 1) # 取feature的targets列,即论文中的r值

        sin_theta = torch.sqrt((1.0 - torch.pow(target_logit, 2)).clamp(0, 1))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        cos_theta_m = cos_theta_m.type_as(target_logit)
        mask = cos_theta > cos_theta_m  # 困难样本
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm) #调整范围[0,pi-m]

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t  # t值需要更新，故需放在forward函数中
        # import pdb; pdb.set_trace()
        cos_theta[mask] = (hard_example * (self.t + hard_example)).type_as(target_logit) # 困难样本 N(t,cosΘ)=coΘ*(t+cosΘ)
        cos_theta.scatter_(1, targets.view(-1, 1).long(), final_target_logit) # one_hot编码
        pred_class_logits = cos_theta * self._s

        # print(pred_class_logits.shape,targets.shape)
        if self.training:
            loss = self.criterion(pred_class_logits,targets)*self.weight_loss
            return loss
        else:
            return pred_class_logits

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self._s, self._m
        )

if __name__ =="__main__":
    pass
    net = ArcfaceLoss_Dropout(in_feat=1644, num_classes=501)
    y = net(torch.ones(5,1644),torch.arange(1, 6).long())
    print(y)

    # x = torch.randn(3,3)
    # y = 1
    # dropout = nn.Dropout(p=0.001)
    # cos_theta = F.linear(input=dropout(F.normalize(x)), weight=F.normalize(x))
    # cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
    # # print(type(cos_theta))
    # print(cos_theta)
    # # print(cos_theta.size())
    # # print(x.size(0))
    # # print(torch.arange(0, x.size(0)))
    # # print(cos_theta[torch.arange(0, x.size(0)), y])
    # target_logit = cos_theta[torch.arange(0, x.size(0)), y].view(-1, 1)
    # sin_theta = torch.sqrt((1.0 - torch.pow(target_logit, 2)).clamp(0, 1))
    # cos_theta_m = target_logit * 1 - sin_theta * 1  # cos(target+margin)
    # cos_theta_m = cos_theta_m.type_as(target_logit)
    # print(cos_theta_m)
    # mask = cos_theta > cos_theta_m
    # print(mask)
    # src = torch.tensor([[1.000], [2.000]])
    # print(src.size())