'''
@Author      : WangPeide
@Contact     : 377079627@qq.com
LastEditors: Please set LastEditors
@Description :
LastEditTime: 2022/5/3 22:06
'''

import pandas as pd
import numpy as np
import glob
import os
# 下面的文件路径需要修改成自己的
path_file = r'F:\PycharmProjects_\Graduation\indexing\index_tools\caltech_split.txt'
# 下面的文件路径需要修改成自己的数据集的位置
root_data =r'F:\PycharmProjects_\Graduation\features\data'
gallery_lst = glob.glob('F:/PycharmProjects_/Graduation/features/data/test_data_A/gallery/*.jpg')
query_lst = glob.glob('F:/PycharmProjects_/Graduation/features/data/test_data_A/query/*.jpg')
gallery_path = ['F:/PycharmProjects_/Graduation/features/data/test_data_A/gallery/'+os.path.basename(i) for i in gallery_lst]
query_lst = ['F:/PycharmProjects_/Graduation/features/data/test_data_A/query/'+os.path.basename(i) for i in query_lst]
print(len(query_lst))
print(len(gallery_lst))
if os.path.exists(path_file):
    os.remove(path_file)

with open(path_file,'w') as f:
    for i in gallery_path:
        f.write('{0} {1}\n'.format(i,1))
    for index,i in enumerate(query_lst):
        if index != len(query_lst)-1:
            f.write('{0} {1}\n'.format(i,0))
        else:
            f.write('{0} {1}'.format(i,0))

with open("caltech_split.txt", 'r') as f:
    lines = f.readlines()
    for line in lines:
        path = line.strip('\n').split(' ')[0]
        is_gallery = line.strip('\n').split(' ')[1]
        if is_gallery == '0':
            src = os.path.join("/data/caltech101/", path)
            print(src)
            dst = src.replace(path.split('/')[0], 'query')
            print(dst)
            dst_index = len(dst.split('/')[-1])
            dst_dir = dst[:len(dst) - dst_index]
            if not os.path.isdir(dst_dir):
                os.makedirs(dst_dir)
            if not os.path.exists(dst):
                os.symlink(src, dst)
        # elif is_gallery == '1':
        #     src = os.path.join("/data/caltech101/", path)
        #     dst = src.replace(path.split('/')[0], 'gallery')
        #     dst_index = len(dst.split('/')[-1])
        #     dst_dir = dst[:len(dst) - dst_index]
        #     if not os.path.isdir(dst_dir):
        #         os.makedirs(dst_dir)
        #     if not os.path.exists(dst):
        #         os.symlink(src, dst)

# python main/split_dataset.py -d /data/caltech101/ -sf caltech_split.txt

