'''
@Author      : WangPeide
@Contact     : 377079627@qq.com
LastEditors: Please set LastEditors
@Description :
LastEditTime: 2022/1/15 11:57
'''

import torch
import torch.nn as nn
from torchvision import models
import torchvision.models.densenet as densenet
from torchsummary import summary
import torch.nn.functional as F
from collections import OrderedDict

# 源码解读
# https://blog.csdn.net/frighting_ing/article/details/121582735
# https://zhuanlan.zhihu.com/p/54767597
# https://zhuanlan.zhihu.com/p/31650605
"""
num_input_features -> bn_size *growth_rate -> growth_rate
concat(num_input_features,growth_rate)
"""
#bottleneck BN+ReLU+Conv(1*1) + BN+ReLU+Conv(3*3)
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))

        self.drop_rate = drop_rate

    def bn_function(self, inputs):

        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)  # 将调用所有add_module方法添加到sequence模块的forwards函数
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,training=self.training)

        return torch.cat([x, new_features], 1)

        # pytorch源码没改通
        # if isinstance(x, torch.Tensor):
        #     prev_features = [x]
        # else:
        #     prev_features = x
        # bottleneck_output = self.bn_function(prev_features)
        # new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        # if self.drop_rate > 0:
        #     new_features = F.dropout(new_features, p=self.drop_rate,
        #                              training=self.training)
        # return new_features

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i+1), layer)

#bottleneck BN+ReLU+Conv(1*1) + AvgPooling(2*2)
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i+1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1: # 如果不是最后一个最后一个block
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i+1), trans)
                num_features = num_features // 2 # 压缩银子theta=0.5 //is 去尾除法 eq:3//2=1

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():  # 判断是哪一类型层，做不同的操作
            if isinstance(m, nn.Conv2d):  # isinstance()会认为子类是一种父类类型，考虑继承关系
                nn.init.kaiming_normal(m.weight.data) # 权重初始化
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        def forward(self, x):
            features = self.features(x)
            out = F.relu(features, inplace=True)
            out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
            out = self.classifier(out)

            return out

def _densenet():
    pass

def build_densenet121(**kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
    return model

def build_densenet169(**kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 24, 32, 32), **kwargs)
    return model

def build_densenet201(**kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)
    return model

def build_densenet264(**kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 64, 48), **kwargs);
    return model

class densenet169(nn.Module):
    def __init__(self,pretrained=True, progress=False, **kwargs):
        super(densenet169,self).__init__()
        backbone = densenet.densenet169(pretrained,progress,**kwargs)

        import re
        pthfile = r'F:\PycharmProjects_\Graduation\features\model\backbone\densenet169-b2777c0a.pth'
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(pthfile)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        backbone.load_state_dict(state_dict)

        self.backbone = backbone.features
    def forward(self,x):
        out = F.relu(self.backbone(x), inplace=True)
        return out # 1664 channel

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # net = _DenseLayer(num_input_features=3, growth_rate=4, bn_size=5, drop_rate=0)
# net = _Transition(num_input_features=64, num_output_features=32).to(device)
# summary(model=net, input_size=(64, 224, 224), batch_size=32, device="cuda")

if __name__ == "__main__":
    pass
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = densenet169().to(device)
    # summary(model=net, input_size=(3, 224, 224), batch_size=128, device="cuda")
    #
    # x = torch.randn(4,3,224,224)
    # y = net(x.cuda())
    # print("y.size(){}".format(y.size()))
    # for index, item in enumerate(net.children()):
    #     print(index, item)


