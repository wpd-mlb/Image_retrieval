'''
@Author      : wpd
@Contact     : 377079627@qq.com
LastEditors: Please set LastEditors
@Description :
LastEditTime: 2021/1/7 18:05
'''

from torch.utils.data import Dataset
# import cv2 as cv
import pandas as pd
import os
import os.path as osp
import numpy as np
import torch
from os.path import  join
import random
from collections import defaultdict
from copy import deepcopy
import matplotlib.pyplot as plt


class load_npyAll(Dataset):
    def __init__(self, image_dir, label_file, transforms=None):
        self.image_dir = image_dir
        self.label = pd.read_csv(label_file, header=None).values
        self.transforms = transforms

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image_path, target = self.label[idx, :].tolist()
        image_path = image_path.replace(".jpg", ".npy")

        image = np.load(osp.join(self.image_dir, image_path))
        sample = {"image": image, "target": target}
        if self.transforms:
            sample = self.transforms(sample)
        image, target = sample['image'], sample['target']

        return image, target


# 改1
class load_npy(Dataset):
    def __init__(self, image_dir, transforms):
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_list = os.listdir(image_dir)

        self.image_list.sort()

    def __len__(self):
        # return len(self.image_list)
        return 1

    def __getitem__(self, idx):
        # image_name = image_name.replace(".jpg", ".npy")

        # print(self.image_dir)# F:\PycharmProjects_\Graduation\features\data\AA_numpy
        # print(image_name)# DIGIX_000000
        # print(osp.join(self.image_dir, image_name))
        images = []
        labels = []
        images_name = []
        data_dict = {}
        i = -1
        for root, _, files in os.walk(self.image_dir):
            # print("root",root)
            # print(files)
            for file in files:
                image = np.load(osp.join(root, file))
                sample = {'image': image}
                if self.transforms:
                    sample = self.transforms(sample)
                image = sample['image']
                # print(image) # torch.Size([3, 512, 512])
                # print(image_name) # 01HMATBIX4CFKDS8.jpg
                # print(image.shape)
                image = image.unsqueeze(0)
                # print(image.shape)
                images.append(image)
                labels.append(i)
                images_name.append(file)
            i=i+1
        image_list = list()
        label_list = list()
        for i in range(len(images)):
            image= images[i]
            image_list.append(image)
        images = torch.cat(image_list, dim=0)
        labels = torch.Tensor(labels)

        print("____",images.shape)
        print("____",labels)
        # return image, image_name
        return images, labels,images_name


# class load_npy(Dataset):
#     def __init__(self, image_dir, transforms):
#         self.image_dir = image_dir
#         self.transforms = transforms
#         self.image_list = os.listdir(image_dir)
#
#         self.image_list.sort()
#
#     def __len__(self):
#         return len(self.image_list)
#
#     def __getitem__(self, idx):
#         image_name = self.image_list[idx]
#         image_name = image_name.replace(".jpg", ".npy")
#
#         image = np.load(osp.join(self.image_dir, image_name))
#         sample = {'image': image}
#         if self.transforms:
#             sample = self.transforms(sample)
#         image = sample['image']
#         image_name = image_name.replace(".npy", ".jpg")
#         return image, image_name

class train_dataset(Dataset):
    def __init__(self, root_dir, images_per_classes, classes_per_minibatch, transforms=None):

        self.transform = transforms  # 变换
        self.classes_per_minibatch = classes_per_minibatch
        self.images_per_classes = images_per_classes
        self.minibatch = classes_per_minibatch * images_per_classes

        filename_filted = 'label.txt'
        lst = os.listdir(root_dir)
        self.num_all_classes = np.sum(list(map(lambda x: x[-4:] != '.txt', lst)))  # 训练集的ID总数

        self.data = [[] for i in range(self.num_all_classes)]
        self.label = []

        file = open(join(root_dir, filename_filted))
        while 1:
            line = file.readline()
            if not line:
                break
            line = line.strip('\n')
            data_l = line.split(',')
            data_npy = data_l[0][:-3] + 'npy'
            self.label.append(int(data_l[1]))
            self.data[int(data_l[1])].append(join(root_dir, data_npy))
        file.close()

        self.steps_all = int(len(self.data) / classes_per_minibatch)  # 501/1
        self.read_order = list(i for i in range(self.num_all_classes))
        # self.read_order = random.sample(range(0, self.num_all_classes),self.num_all_classes) # 读入顺序打乱
        # random.sample(list,num) 实现返回list中num个随机数字，返回的是列表

    def shuffle(self):  # 每运行完一个epoch,运行该函数打乱顺序
        self.read_order = random.sample(range(0,self.num_all_classes),self.num_all_classes)
        # for class_id in range(len(self.data)):
        #     self.data[class_id]=random.shuffle(self.data[class_id])

    def get_item(self,class_id,img_id):
        img = np.load(join(self.data[class_id][img_id])) # np载入class_id类，随机的一张img_id照片
        sample = {"image": img}
        if self.transform:
            sample = self.transform(sample)  # 对样本进行变换
        img = sample['image']

        # debug
        img=torch.tensor(img)
        img = img.unsqueeze(0)
        # print(img.shape)
        return img

    def __len__(self):  # 获取总的mini_batch数目
        # return len(self.label)
        return self.steps_all

    def __getitem__(self, step):  # 获取第step个minibatch
        if step > self.steps_all - 1:
            print('step_train out of size')
            return

        # print(len(self.read_order))
        class_ids = self.read_order[step * self.classes_per_minibatch:(step + 1) * self.classes_per_minibatch]
        # class_ids=random.sample(range(0,self.num_all_classes),self.classes_per_minibatch)

        start = True
        labels = []

        for class_id in class_ids:  # 取出单元素列表中的值
            num = min(self.images_per_classes, len(self.data[class_id]))
            # if num<2:
            #     # print('第{}类图片数目不够'.format(class_id))
            #     continue
            while (num < 2):
                class_id = np.random.choice(501, 1)[0]
                # random.choice(n,nn) [0,n)中随机输出nn个随机数，返回的是列表，切片
                num = len(self.data[class_id])
            img_ids = np.random.choice(len(self.data[class_id]), self.images_per_classes)
            for img_id in img_ids:

                img_tmp = self.get_item(class_id, img_id)  # tensor(batch size, height, width, channels)
                labels.append(class_id)
                if start:
                    imgs = img_tmp.detach().clone()  # 分离 不需要梯度
                    start = False
                else:
                    imgs = torch.cat((imgs, img_tmp), dim=0)  # 将batch拼接

        labels = torch.tensor(labels)
        labels = labels.int()

        # with open('/home/yufei/HUW/models/trip_loss/log/tmp_data.txt',"a") as file:   #”w"代表着每次运行都覆盖内容
        #     file.write('===========================================\n')
        return imgs, labels


if __name__ == '__main__':
    pass
    dataset = train_dataset(root_dir=r"F:\PycharmProjects_\Graduation\features\data\train_data_resize512",
                  images_per_classes=4,
                  classes_per_minibatch=1,)
    # with open('F:\\PycharmProjects_\\Graduation\\features\\data\\' + 'shape.txt', 'a') as f:
            # f.write(str(a.shape) + '\n' + str(b) + '\n')
    # f.close()
    # plt.imshow(a[0].numpy())
    # plt.show()


    # dataset = load_npyAll(image_dir=r"F:\PycharmProjects_\Graduation\features\data\train_data_resize512",
    #                       label_file=r"F:\PycharmProjects_\Graduation\features\data\train_data_resize512\label.txt", )
    a,b = dataset.__getitem__(46)
    print(b)
    a=a[0,:,:,:]
    print(a.size())
    plt.imshow(a)
    plt.show()

    # label = pd.read_csv(r"F:\PycharmProjects_\Graduation\features\data\train_data\label.txt", header=None).values
    # print(type(label))
    # print(len(label))
