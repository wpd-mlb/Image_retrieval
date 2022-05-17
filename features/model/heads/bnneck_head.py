'''
@Author      : WangPeide
@Contact     : 377079627@qq.com
LastEditors: Please set LastEditors
@Description :
LastEditTime: 2022/1/17 22:01
'''

import math
import torch
import torch.nn as nn


class BNneckHead(nn.Module):
    def __init__(self, in_feat, num_classes):
        super(BNneckHead, self).__init__()
        self.bnneck = nn.BatchNorm2d(in_feat)
        self.bnneck.apply(weights_init_kaiming)
        self.bnneck.bias.requires_grad_(False)  # no shift

    def forward(self, features):
        return self.bnneck(features)[..., 0, 0] # [b,c,h,w] ->[b,c]取左上角元素


class BNneckHead_Dropout(nn.Module):
    def __init__(self, in_feat, num_classes, dropout_rate=0.15):
        super(BNneckHead_Dropout, self).__init__()
        self.bnneck = nn.BatchNorm2d(in_feat)
        self.bnneck.apply(weights_init_kaiming)
        self.bnneck.bias.requires_grad_(False)  # no shift
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, features):
        return self.dropout(self.bnneck(features)[..., 0, 0])

def weights_init_kaiming(m):
    classname = m.__class__.__name__  # weights_init_kaiming
    # print("classname:{}".format(classname))
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("Arcface") != -1 or classname.find("Circle") != -1:
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

if __name__ == "__main__":
    pass
    # x = torch.randn(1,3,2,2)
    # net = BNneckHead(in_feat=3, num_classes=3096)
    # y = net(x)
    # print(y.size())
    # print(y)

    # x = torch.randn(1, 3, 3, 3)
    # print(x)
    # y = x[..., 0, 0]
    # print(y)
    # print(y.shape)

    # w = torch.empty(3, 5)
    # print(w)
    # nn.init.normal_(w)

