'''
@Author      : WangPeide
@Contact     : 377079627@qq.com
LastEditors: Please set LastEditors
@Description :
LastEditTime: 2022/4/12 10:12
'''

import pandas as pd
import os
from os.path import join,dirname,realpath
import pickle
import argparse
import numpy as np
from shutil import copyfile

def parse_args():
    parser = argparse.ArgumentParser(description='A tool box for deep learning-based image retrieval')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--here_dir', default="F:/PycharmProjects_/Graduation/indexing/features", type=str)
    parser.add_argument('--mode', default='concat', type=str)
    parser.add_argument('--json_dir', nargs='+')
    # default = "F:/PycharmProjects_/Graduation/features/exp/hw_json",
    args = parser.parse_args()
    return args

# python ./index_tools/trans_json_to_here.py \
# --here_dir ./features/hw2_json \
# --json_dir /home/LinHonghui/Project/v2/features/exp/20201102_testC_efficientb5_80


def query(data_root,args):
    #query
    query_data_json=[]
    dict_json = args.json_dir

    for dict_json in args.json_dir:
        # print("\n=== {} ===".format(dict_json))
        tmp = join(dict_json, 'query.json')
        load_f = open(tmp, "rb")
        df = pickle.load(load_f)
        num_images = len(df['fname']) # 500
        query_data_json.append(df['data']) # df['data'].shape ---->(500,1664)
    query_data_json=np.array(query_data_json).astype(np.float32)#[1,9600,10240]  # [1,500,1664]
    # print(query_data_json.shape)
    num_dir = len(args.json_dir)
    # print(num_dir)  # 1  F:/PycharmProjects_/Graduation/features/exp/20220405_tag_example_130/

    query_dir = join(args.here_dir, 'query')
    if not os.path.exists(query_dir):
        os.makedirs(query_dir)
    query_name = join(query_dir, 'part_0.json')

    dist_query = {}
    dist_query['nr_class'] = num_images
    dist_query['path_type'] = 'absolute_path'
    dist_query['info_dicts'] = []
    # print(type(df["fname"]), query_data_json.shape)
    for i in range(len(df["fname"])):
        img_name = df["fname"][i][0]
        dict_tmp = {}
        dict_tmp['path'] = join(data_root,'test_data_A', 'gallery', img_name)

        #测试mAP代码
        # dict_tmp['path'] = join(data_root, 'AA', 'gallery', img_name)
        # dict_tmp['label_idx'] = df["label"][i]

        dict_tmp['label'] = img_name
        dict_tmp['query_name'] = img_name
        dict_tmp['idx'] = i
        dict_tmp['feature'] = {}
        print(dict_tmp)
        feature_lst_1 = []
        if args.mode == 'concat':
            # print(query_data_json[j,i,:].shape,num_dir,len(df["fname"]),dict_tmp['path'])
            # assert 1==0
            feature_lst_1 = feature_lst_1 + query_data_json[0, i, :].tolist() # 1664
        dict_tmp['feature']['f1'] = feature_lst_1
        dist_query['info_dicts'].append(dict_tmp)
        if (i % 100 == 0):
            print('query:{}/{}'.format(i + 1, num_images))

    with open(query_name, "wb") as f:
        pickle.dump(dist_query, f)


def gallery(data_root,args):
    # gallery
    gallery_data_json = []
    for dict_json in args.json_dir:
        tmp = join(dict_json, 'gallery.json')
        # df=np.load(tmp)
        load_f = open(tmp, "rb")
        df = pickle.load(load_f)
        num_images = len(df['fname'])
        gallery_data_json.append(df['data'])

    gallery_data_json = np.array(gallery_data_json).astype(np.float32)
    num_dir = len(args.json_dir)

    gallery_dir = join(args.here_dir, 'gallery')
    if not os.path.exists(gallery_dir):
        os.makedirs(gallery_dir)
    gallery_name = join(gallery_dir, 'part_0.json')

    dist_gallery = {}

    dist_gallery['nr_class'] = num_images
    dist_gallery['path_type'] = 'absolute_path'
    dist_gallery['info_dicts'] = []
    for i in range(len(df["fname"])):
        img_name = df["fname"][i][0]
        dict_tmp = {}
        dict_tmp['path'] = join(data_root,'test_data_A', 'gallery', img_name)

        # 测试mAP代码
        # dict_tmp['path'] = join(data_root, 'AA', 'gallery', img_name)
        # dict_tmp['label_idx'] = df["label"][i]

        dict_tmp['label'] = img_name
        dict_tmp['idx'] = i
        dict_tmp['feature'] = {}
        print(dict_tmp)
        feature_lst_1 = []
        for j in range(num_dir):
            if args.mode == 'concat':
                # print(query_data_json[j,i,:].shape,num_dir,len(df["fname"]),dict_tmp['path'])
                # assert 1==0
                feature_lst_1 = feature_lst_1 + gallery_data_json[j, i, :].tolist()

        dict_tmp['feature']['f1'] = feature_lst_1
        dist_gallery['info_dicts'].append(dict_tmp)
        if (i % 100 == 0):
            print('gallery:{}/{}'.format(i + 1, num_images))
        i = i + 1

    with open(gallery_name, "wb") as f:
        pickle.dump(dist_gallery, f)




def main():
    # data_root='/home/yufei/HUW2/data'
    data_root =r'F:\PycharmProjects_\Graduation\features\data'

    # init args
    args = parse_args()

    # 改pyreri的 main/make_data_json.py code ——————没改成功
    # dataset_path = "F:/PycharmProjects_/Graduation/features/data/test_data_A/query"
    # save_path = "F:/PycharmProjects_/Graduation/indexing/data_jsons/query.json"
    # label_list = list()
    # info_dicts = list()
    # for root, _, files in os.walk(dataset_path):
    #     print("====================")
    #     print("现在的目录：" + root)
    #     print("该目录下包含的文件：" + str(files))
    #     i:int = 0
    #     for file in files:
    #         info_dict = dict()
    #         info_dict['path'] = os.path.join(root, file)
    #         # print(file)
    #         info_dict['label'] = file
    #         info_dict['query_name'] = file
    #         info_dict['idx'] = i
    #         i = i+1
    #         print(info_dict, "\t")
    #         info_dicts += [info_dict]
    #     f = open(save_path, "wb")
    #     pickle.dump({'nr_class': i, 'path_type': 'absolute_path', 'info_dicts': info_dicts}, f)
    #     f.close()

    query(data_root,args)
    gallery(data_root,args)



if __name__ == '__main__':
    main()

# Terminal中执行
# python trans_json_to_here.py --json_dir F:/PycharmProjects_/Graduation/features/exp/hw_json
# {'path': 'F:\\PycharmProjects_\\Graduation\\features\\data\\test_data_A\\query\\2MLEX7ICP9K6QVFN.jpg',
# 'label': '2MLEX7ICP9K6QVFN.jpg',
# 'query_name': '2MLEX7ICP9K6QVFN.jpg',
# 'idx': 499,
# 'feature': {}}
