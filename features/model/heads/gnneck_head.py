'''
@Author      : WangPeide
@Contact     : 377079627@qq.com
LastEditors: Please set LastEditors
@Description :
LastEditTime: 2022/1/17 14:34
'''

import math
import torch
import torch.nn as nn
# from utils import weights_init_kaiming

class GNneckHead(nn.Module):
    def __init__(self, num_channels, num_groups=32, num_classes=3097):
        super(GNneckHead, self).__init__()
        self.gnneck = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        self.gnneck.apply(weights_init_kaiming)
        self.gnneck.bias.requires_grad_(False)  # no shift

    def forward(self, features):
        return self.gnneck(features)[..., 0, 0]

# for test
@torch.no_grad()
def init_weights(m):
    print('mm:{}'.format(m))
    if type(m) == nn.Linear:
        m.weight.fill_(1.1)
        print(m.weight)

def weights_init_kaiming(m):
    classname = m.__class__.__name__  # weights_init_kaiming
    print("classname{}".format(classname))
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
    # net = nn.Sequential(nn.Linear(3, 4))
    # net.apply(init_weights)

    # weights_init_kaiming("a")






