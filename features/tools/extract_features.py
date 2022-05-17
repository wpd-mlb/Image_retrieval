'''
@Author      : WangPeide
@Contact     : 377079627@qq.com
LastEditors: Please set LastEditors
@Description :
LastEditTime: 2022/2/13 13:02
'''
import sys
sys.path.append("..")
from data.dataloader import make_dataloader
from configs import merage_from_arg,load_arg
from model import build_model
from argparse import ArgumentParser
import torch
import torch.nn as nn
from utils import get_free_device_ids
import copy
import datetime
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle

# cudnn实现网络的加速
torch.backends.cudnn.benchmark = True

def dict_to_file(file_path,dict_data):
    f = open(file_path, "wb")
    pickle.dump(dict_data, f)
    f.close()

# def ADD_TTA_inference_extract_features(cfg,model,dataloader,device_ids,feature_type="after"):
#     print("------- start -------")
#     master_device = device_ids[0]
#     model.eval()
#     features_list = []
#     label_list = []
#     filename_list = []
#     data_dict = {}
#     with torch.no_grad():
#         for image, label,filename in tqdm(dataloader):  # filename: tuple(batch_filename)
#             image = image.squeeze(0)
#             label = label.squeeze(0)
#             print(image.shape,label.shape)
#             # test
#             # print("batch的shape：", image.shape, type(filename))
#             # import matplotlib.pyplot as plt
#             # image=image[0,:,:,:]
#             # print(image.size())
#             # image = image.permute(1, 2, 0)  # add
#             # print(image.size())
#             # plt.imshow(image) # torch.Size([512, 512, 3])
#             # plt.show()
#             # print("original",image.shape)
#             image = image.cuda(master_device)
#
#             features_1 = model(image.clone(), extract_features_flag=True, feature_type=feature_type)
#             features_1 = features_1.cpu().data.numpy().astype(np.float32)
#             # print("change_0", features_1.shape)
#             # 水平
#             features_2 = model(torch.flip(image.clone(), [-1]), extract_features_flag=True, feature_type=feature_type)
#             features_2 = features_2.cpu().data.numpy().astype(np.float32)
#
#             # 垂直
#             features_3 = model(torch.flip(image.clone(), [-2]), extract_features_flag=True, feature_type=feature_type)
#             features_3 = features_3.cpu().data.numpy().astype(np.float32)
#
#             # 顺时针90
#             features_4 = model(torch.flip(image.clone().transpose(2, 3), [-1]), extract_features_flag=True,
#                                feature_type=feature_type)
#             features_4 = features_4.cpu().data.numpy().astype(np.float32)
#
#             # 逆时针90
#             features_5 = model(torch.flip(image.clone().transpose(2, 3), [-2]), extract_features_flag=True,
#                                feature_type=feature_type)
#             features_5 = features_5.cpu().data.numpy().astype(np.float32)
#
#             features = features_1 + features_2 + features_3 + features_4 + features_5
#             # print("change_1",features.shape)
#             # import pdb; pdb.set_trace()
#             features_list.append(features)
#             filename_list.append(filename)
#             label_list.append(label)
#
#         np_filename = np.concatenate(filename_list)
#         np_label = np.concatenate(label_list)
#         np_features = np.concatenate(features_list).astype(np.float32)
#         # print("end_1",len(features_list)) # 17
#         # print("end_2", np_filename.shape) # (500,)
#         # print("end_3", np_features.shape) # (500, 1664)
#         # exit()
#     data_dict['fname'] = np_filename
#     data_dict['data'] = np_features
#     data_dict['label'] = np_label
#     print(len(np_filename),len(np_label))
#     print(np_filename)
#     return data_dict

def ADD_TTA_inference_extract_features(cfg, model, dataloader, device_ids, feature_type="after"):
    print("------- start -------")
    master_device = device_ids[0]
    model.eval()
    features_list = []
    filename_list = []
    data_dict = {}
    with torch.no_grad():
        for image, filename in tqdm(dataloader):  # filename: tuple(batch_filename)
            image = image.cuda(master_device)

            features_1 = model(image.clone(), extract_features_flag=True, feature_type=feature_type)
            features_1 = features_1.cpu().data.numpy().astype(np.float32)

            # 水平
            features_2 = model(torch.flip(image.clone(), [-1]), extract_features_flag=True, feature_type=feature_type)
            features_2 = features_2.cpu().data.numpy().astype(np.float32)

            # 垂直
            features_3 = model(torch.flip(image.clone(), [-2]), extract_features_flag=True, feature_type=feature_type)
            features_3 = features_3.cpu().data.numpy().astype(np.float32)

            # 顺时针90
            features_4 = model(torch.flip(image.clone().transpose(2, 3), [-1]), extract_features_flag=True,
                               feature_type=feature_type)
            features_4 = features_4.cpu().data.numpy().astype(np.float32)

            # 逆时针90
            features_5 = model(torch.flip(image.clone().transpose(2, 3), [-2]), extract_features_flag=True,
                               feature_type=feature_type)
            features_5 = features_5.cpu().data.numpy().astype(np.float32)

            features = features_1 + features_2 + features_3 + features_4 + features_5
            # import pdb; pdb.set_trace()
            features_list.append(features)
            filename_list.append(filename)
        np_filename = np.concatenate(filename_list)
        np_features = np.concatenate(features_list).astype(np.float32)
    data_dict['fname'] = np_filename
    data_dict['data'] = np_features
    return data_dict

if __name__ == "__main__":
    # 若更新了load_arg函数，需要对应更新merage_from_arg()
    arg = vars(load_arg())
    if arg['load_path'] != None: #优先级：arg传入命令 >model中存的cfg > config_file
        state_dict = torch.load(arg['load_path'],map_location='cpu')
        if 'cfg' in state_dict.keys():
            cfg = state_dict['cfg']
    for key, value in arg.items():
        print('{key}:{value}'.format(key=key, value=value))

    # 待修改
    config_file = arg["CONFIG_FILE"]
    config_file = config_file.replace("../","").replace(".py","").replace('/','.')
    exec(r"from {} import config as cfg".format(config_file))
    cfg = merage_from_arg(cfg, arg)

    # save_dir
    # curr_time = datetime.datetime.now()
    # time_str = datetime.datetime.strftime(curr_time, '%Y%m%d_')
    # save_dir = os.path.join(cfg['save_dir'], time_str + cfg['tag'])
    # cfg['save_dir'] = save_dir
    # if not os.path.isdir(save_dir):
    #     os.makedirs(save_dir)
    # print("Save_dir :", save_dir)

    # model_tag = (os.path.split(arg['load_path'])[1]).split('.')[0]
    # save_dir = os.path.join(r'../exp/', time_str + model_tag)
    # if not os.path.isdir(save_dir):
    #     os.makedirs(save_dir)
    # print("Save_dir :", save_dir)


    # load `model` & `dataloader`
    model = build_model(cfg, pretrain_path=arg['load_path'])
    # get free_device
    free_device_ids = get_free_device_ids()
    max_num_devices = cfg['max_num_devices']
    if len(free_device_ids) >= max_num_devices:
        free_device_ids = free_device_ids[:max_num_devices]
    # print(free_device_ids)

    # data parallel
    master_device = free_device_ids[0]
    model = nn.DataParallel(model, device_ids=free_device_ids).cuda(master_device)

    # save_dir
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y%m%d_')
    model_tag = (os.path.split(arg['load_path'])[1]).split('.')[0]
    save_dir = os.path.join(r'../exp/', time_str + model_tag)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    print("Save_dir :", save_dir)


    dataloader = make_dataloader(cfg['query_pipeline'])
    dict_query = ADD_TTA_inference_extract_features(cfg,model,dataloader,device_ids=free_device_ids)
    # images, labels = next(iter(dataloader))  # [batch_size,channel,28,28] [batch_size,1]
    # print("batch的shape：",images.shape, labels.shape)
    dict_to_file(os.path.join(save_dir,"query.json"),dict_query)

    dataloader = make_dataloader(cfg['gallery_pipeline'])
    # images, labels = next(iter(dataloader))  # [batch_size,channel,28,28] [batch_size,1]
    # print("batch的shape：", images.shape, labels.shape)
    dict_gallery = ADD_TTA_inference_extract_features(cfg,model,dataloader,device_ids=free_device_ids)
    dict_to_file(os.path.join(save_dir,"gallery.json"),dict_gallery)