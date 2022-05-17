'''
@Author      : WangPeide
@Contact     : 377079627@qq.com
LastEditors: Please set LastEditors
@Description :
LastEditTime: 2022/4/17 10:56
'''
import os
import random

from PIL import Image
import pandas as pd
import os
import random
from shutil import copyfile
import shutil

submit_file_path='F:/PycharmProjects_/Graduation/indexing/result/submission.csv'
test_data_dir="F:/PycharmProjects_/Graduation/features/data/test_data_A"
out_dir='F:/PycharmProjects_/Graduation/indexing/visual'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
else:
    shutil.rmtree(out_dir)
    os.mkdir(out_dir)

sf = pd.read_csv(submit_file_path, header=None)

n=40
sample_lst=random.sample(range(0,len(sf)),n)#随机抽10个
print(sample_lst)
i=1
for num in sample_lst:

    d_dir = os.path.join(out_dir, str(i))
    os.makedirs(d_dir)

    query_img_path = os.path.join(test_data_dir, "query", sf[0][num])
    copyfile(query_img_path, os.path.join(d_dir, '0_' + str(i) + "query.jpg"))

    gallery_img_path = os.path.join(test_data_dir, "gallery", sf[1][num][1:])
    copyfile(gallery_img_path, os.path.join(d_dir, str(i) + "gallery" + str(0) + ".jpg"))

    for j in range(2, 11):
        gallery_img_name = sf[j][num]
        # gallery_img_name=gallery_img_name.replace(' ','')
        gallery_img_name = gallery_img_name.replace('{', '')
        gallery_img_name = gallery_img_name.replace('}', '')
        gallery_img_path = os.path.join(test_data_dir, "gallery", gallery_img_name)
        copyfile(gallery_img_path, os.path.join(d_dir, str(i) + "gallery" + str(j) + ".jpg"))

    i = i + 1