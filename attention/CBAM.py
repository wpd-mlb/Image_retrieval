'''
@Author      : WangPeide
@Contact     : 377079627@qq.com
LastEditors: Please set LastEditors
@Description :
LastEditTime: 2022/5/14 23:49
'''
"""
Author: yida
Time is: 2021/11/21 11:40 
this Code: 实现CBAM模块
"""
# https://blog.csdn.net/qq_42178122/article/details/122146994?utm_term=pytorch%20%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6&utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~sobaiduweb~default-0-122146994-null-null&spm=3001.4430
import os
import torch
import torch.nn as nn
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.Cam = ChannelAttentionModul(in_channel=in_channel)  # 通道注意力模块
        self.Sam = SpatialAttentionModul(in_channel=in_channel)  # 空间注意力模块

    def forward(self, x):
        x = self.Cam(x)
        x = self.Sam(x)
        return x


class ChannelAttentionModul(nn.Module):  # 通道注意力模块
    def __init__(self, in_channel, r=0.5):  # channel为输入的维度, r为全连接层缩放比例->控制中间层个数
        super(ChannelAttentionModul, self).__init__()
        # 全局最大池化
        self.MaxPool = nn.AdaptiveMaxPool2d(1)

        self.fc_MaxPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # 全局均值池化
        self.AvgPool = nn.AdaptiveAvgPool2d(1)

        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1.最大池化分支
        max_branch = self.MaxPool(x)
        # 送入MLP全连接神经网络, 得到权重
        max_in = max_branch.view(max_branch.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)

        # 2.全局池化分支
        avg_branch = self.AvgPool(x)
        # 送入MLP全连接神经网络, 得到权重
        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)

        # MaxPool + AvgPool 激活后得到权重weight
        weight = max_weight + avg_weight
        weight = self.sigmoid(weight)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        h, w = weight.shape
        # 通道注意力Mc
        Mc = torch.reshape(weight, (h, w, 1, 1))

        # 乘积获得结果
        x = Mc * x

        return x


class SpatialAttentionModul(nn.Module):  # 空间注意力模块
    def __init__(self, in_channel):
        super(SpatialAttentionModul, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x维度为 [N, C, H, W] 沿着维度C进行操作, 所以dim=1, 结果为[N, H, W]
        MaxPool = torch.max(x, dim=1).values  # torch.max 返回的是索引和value， 要用.values去访问值才行！
        AvgPool = torch.mean(x, dim=1)

        # 增加维度, 变成 [N, 1, H, W]
        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)

        # 维度拼接 [N, 2, H, W]
        x_cat = torch.cat((MaxPool, AvgPool), dim=1)  # 获得特征图

        # 卷积操作得到空间注意力结果
        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)

        # 与原图通道进行乘积
        x = Ms * x

        return x


if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    img = cv2.imread(r"F:\PycharmProjects_\Graduation\1111.png")
    # b, g, r = cv2.split(img)
    # img = cv2.merge((r, g, b))
    # print(img.shape)
    img_1 = cv2.resize(img, (256, 256))
    # print(img_1.shape)
    x = torch.tensor(img_1).float()
    # x = x.unsqueeze(0)
    x = x.unsqueeze(0)
    # print(x.shape)

    plt.figure()
    images = x[0, :, :, :]
    # images = images.permute(1, 2, 0)  # add
    # print(images.size())
    plt.imshow(images.cpu().numpy().astype(np.uint8))
    plt.show()

    inputs = x.permute(0, 3,1, 2)
    # inputs = torch.randn(1, 3, 224, 224)
    #
    model = CBAM(in_channel=3)  # CBAM模块, 可以插入CNN及任意网络中, 输入特征图in_channel的维度
    # print(model)
    outputs = model(inputs)

    # print("输入维度:", inputs.shape)
    # print("输出维度:", outputs.shape)

    plt.figure()
    outputs = outputs[0, :, :, :]
    outputs = outputs.permute(1, 2, 0)  # add
    print(outputs.size())
    print(outputs)
    plt.imshow(outputs.cpu().detach().numpy().astype(np.uint8))
    plt.show()
