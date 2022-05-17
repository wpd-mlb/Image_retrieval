'''
@Author      : WangPeide
@Contact     : 377079627@qq.com
LastEditors: Please set LastEditors
@Description :
LastEditTime: 2022/1/16 16:50
'''

import torch
import torch.nn.functional as F
from torch import nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class AdaptiveAvgPool2d(nn.Module):
    def __init__(self,output_size=(1,1),flatten=False):
        super(AdaptiveAvgPool2d, self).__init__()
        self.flatten = flatten
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=output_size)
    def forward(self, x):
        x = self.avgpool(x)
        if self.flatten:
            x = torch.flatten(x, 1)
        return x

class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(self):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.avgpool = FastGlobalAvgPool2d()

    def forward(self, x):
        x_avg = self.avgpool(x, self.output_size)
        x_max = F.adaptive_max_pool2d(x, 1)
        x = x_max + x_avg
        return x


class FastGlobalAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)

#GeM pooling
class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm=3, output_size=(1, 1), eps=1e-6, *args, **kwargs):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'

class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


if __name__ == "__main__":
    pass
    # print(GeneralizedMeanPooling())
    # print(torch.ones(1))
    x = torch.randn(32,3,224,224)
    net = GeneralizedMeanPoolingP()
    y = net(x)
    print(y.size())
    # out = torch.flatten(x, 2)
    # print(out.size())