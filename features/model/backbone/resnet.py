'''
@Author      : WangPeide
@Contact     : 377079627@qq.com
LastEditors: Please set LastEditors
@Description :
LastEditTime: 2022/1/12 20:45
'''

import math
import torch
import torch.nn as nn
from torchsummary import summary
import torchvision.models.resnet as resnet
from torchvision import models
# from .utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url

# 源码解读
# https://blog.csdn.net/m0_50127633/article/details/117200212
# https://www.jianshu.com/p/90d61f53d15d
# https://www.codeleading.com/article/8732963303/
# https://cloud.tencent.com/developer/article/1608203
# https://zhuanlan.zhihu.com/p/74230238

# 调试技巧PrintLayer ——pytorch论坛
class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)  # print(x.shape)
        return x

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class base_resnet(nn.Module):
    def __init__(self):
        super(base_resnet, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.load_state_dict(torch.load('./resnet50-19c8e357.pth'))
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 10000)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.fc(x)
        return x

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# print(type((conv3x3(3, 64, 1))))

#这个实现的是两层的残差块，用于resnet18/34
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None: #当连接的维度不同时，使用1*1的卷积核将低维转成高维，然后才能进行相加
            identity = self.downsample(x)

        out += identity #实现H(x)=F(x)+x或H(x)=F(x)+Wx
        out = self.relu(out)

        return out

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net = BasicBlock(inplanes=64,planes=64).to(device)
# summary(model=net, input_size=(64, 56, 56), batch_size=128,device="cuda")

#这个实现的是三层的残差块，用于resnet50/101/152
class Bottleneck(nn.Module):
    expansion = 4
# 64 128 256 512
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x) #当连接的维度不同时，使用1*1的卷积核将低维转成高维，然后才能进行相加

        out += identity #实现H(x)=F(x)+x或H(x)=F(x)+Wx
        out = self.relu(out)

        return out
#input[b, 256, 56, 56] output[b,512, 28, 28]
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net = Bottleneck(inplanes=64,planes=64).to(device)
# summary(model=net, input_size=(64, 56, 56), batch_size=128,device="cuda")

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(2048, 10)
        self.layer = nn.Sequential(
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # out = self.avgpool(out)
        # out = out.reshape(x.shape[0], -1)
        # out = self.fc(out)
        return x

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net = Test().to(device)
# summary(model=net, input_size=(3, 224, 224), batch_size=128,device="cuda")

class ResNet(nn.Module):
    """
    参数block指明残差块是两层或三层，
    参数layers指明每个卷积层需要的残差块数量，
    num_classes指明分类数，
    zero_init_residual是否初始化为0
    """
    def __init__(self, block, layers, num_classes=1000,zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        # 网络输入部分
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 中间卷积部分
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 平均池化和全连接层
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # 对卷积层进行初始化
        # https://www.codeleading.com/article/8732963303/
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # kaiming高斯初始化，目的是使得Conv2d卷积层反向传播的输出的方差都为1
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # 初始化m.weight，即gamma的值为1；m.bias即beta的值为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  #input[b, 64, 56, 56] output[b,256, 56, 56]
        x = self.layer2(x) #input[b, 256, 56, 56] output[b,512, 28, 28]
        x = self.layer3(x) #input[b,512, 28, 28] output[b,1024, 14, 14]
        x = self.layer4(x) #input[b,1024, 14, 14] output[b,2048, 7, 7]

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = x.view(x.size(0), -1)
        # x = x.view(x.size(0), x.size(1))
        x = self.fc(x)
        return x

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)
# summary(model=net, input_size=(3, 224, 224), batch_size=128,device="cuda")

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    else:
        pass
        # model.load_state_dict(torch.load('./resnet50-19c8e357.pth'))
    return model

def build_resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def build_resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net = build_resnet50().to(device)
# summary(model=net, input_size=(3, 224, 224), batch_size=128,device="cuda")

# https://blog.csdn.net/chen_xinjia/article/details/80273592
class resnet50(nn.Module):
    def __init__(self,pretrained=False,progress=True,**kwargs):
        super(resnet50,self).__init__()
        backbone = build_resnet50(pretrained=pretrained,progress=progress,**kwargs)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, 500)  # 3097
    def forward(self,x):
        x = self.backbone(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        return x

class resnet101(nn.Module):
    def __init__(self,pretrained=False,progress=True,**kwargs):
        super(resnet101,self).__init__()
        backbone = build_resnet101(pretrained=pretrained,progress=progress,**kwargs)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
    def forward(self,x):
        return self.backbone(x)



if __name__ == '__main__':
    pass
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = resnet50().to(device)
    # summary(model=net, input_size=(3, 224, 224), batch_size=32, device="cuda")

    # pretrained=True就可以使用预训练的模型
    # net = models.resnet50(pretrained=False)
    # pthfile = r'./resnet50-19c8e357.pth'
    # net.load_state_dict(torch.load(pthfile))
    # print(net)



