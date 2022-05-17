'''·~~~~~~~~~~~`
@Author      : WangPeide
@Contact     : 377079627@qq.com
LastEditors: Please set LastEditors
@Description :
LastEditTime: 2021/12/31 21:35
'''


import sys
sys.path.append('..')
from data.dataloader import make_dataloader
from configs import merage_from_arg,load_arg
from model import build_model
from solver import make_optimizer,wrapper_lr_scheduler
from argparse import ArgumentParser
from engine import do_train
from configs.base import config as cfg
import torch.nn as nn
import torch
from utils import get_free_device_ids
import copy
import datetime
import os




if __name__ == "__main__":
    pass
    # 若更新了load_arg函数，需要对应更新merage_from_arg()
    arg = vars(load_arg())  # 返回attribute和attribute values的dict对象
    # print(arg)
    for key, value in arg.items():
        print('{key}:{value}'.format(key=key, value=value))

    # 待修改
    config_file = arg["CONFIG_FILE"]  # config_file
    default = str("F:\\PycharmProjects_\\Graduation\\features\\data")
    # config_file = default.replace("../","").replace(".py","").replace('/','.')

    # exec(r"from {} import config as cfg".format(config_file))  # exec执行 字符串中的python代码

    # if arg['MODEL.LOAD_PATH'] != None: #优先级：arg传入命令 >model中存的cfg > config_file
    #     cfg = torch.load(arg['MODEL.LOAD_PATH'])['cfg']\
    cfg = merage_from_arg(cfg,arg)
    # print(cfg)
    cfg_copy = copy.deepcopy(cfg)

    train_dataloader = make_dataloader(cfg['train_pipeline'])
    images, labels = next(iter(train_dataloader))  # [batch_size,channel,28,28] [batch_size,1]
    # print("batch的shape：",images.shape, labels.shape)
    print(labels)
    # import matplotlib.pyplot as plt
    # images=images[0,:,:,:]
    # images = images.permute(1, 2, 0)  # add
    # print(images.size())
    # plt.imshow(images)
    # plt.show()

    curr_time = datetime.datetime.now()
    print(curr_time)
    time_str = datetime.datetime.strftime(curr_time,'%Y%m%d_')
    save_dir = os.path.join(cfg['save_dir'],time_str+cfg['tag'])
    log_dir = os.path.join(cfg['log_dir'],"log_"+time_str+cfg['tag'])
    cfg['save_dir'] = save_dir
    cfg['log_dir'] = log_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    print("Save_dir :",save_dir)
    print("Log_dir :", log_dir)

    # import pdb; pdb.set_trace()
    model = build_model(cfg, pretrain_path=arg['load_path'])

    optimizer = make_optimizer(cfg['optimizer'], model)
    lr_scheduler = wrapper_lr_scheduler(cfg['lr_scheduler'], optimizer)

    if arg['device']: # 传入命令指定 device id
        free_device_ids = arg['device']
    else:
        free_device_ids = get_free_device_ids()

    max_num_devices = cfg['max_num_devices']
    if len(free_device_ids)>=max_num_devices:
        free_device_ids = free_device_ids[:max_num_devices]

    master_device = free_device_ids[0]
    print("master_device:",master_device)
    model.cuda(master_device)
    model = nn.DataParallel(model, device_ids=free_device_ids).cuda(master_device)
    #data parallel

    if cfg['enable_backends_cudnn_benchmark']:
        print("enable backends cudnn benchmark")
        torch.backends.cudnn.benchmark = True
        # cudnn实现网络的加速

    cfg_copy['save_dir'] = save_dir  # 更新存储目录
    cfg_copy['log_dir'] = log_dir  # 更新存储目录
    # import pdb; pdb.set_trace()
    # print(len(train_dataloader))

    # do_train(cfg=cfg_copy, model=model, train_loader=train_dataloader, val_loader=None, optimizer=optimizer,
    #          scheduler=lr_scheduler, metrics=None, device=free_device_ids)

    # model_parameters = [x for x in model.parameters()]
    # print(model_parameters)

    from tqdm import tqdm
    model.eval()
    num_correct = 0
    num_example = 0
    torch.cuda.empty_cache()
    with torch.no_grad():
        for image, target in tqdm(train_dataloader):
            print("value的shape：",image.shape, target.shape)
            image, target = image.to(master_device), target.to(master_device)

            import matplotlib.pyplot as plt
            plt.figure()
            for i in range(len(image)):
                images = image[i, :, :, :]
                images = images.permute(1, 2, 0)  # add
                ax = plt.subplot(2, 2, i + 1)
                print(images.size())
                plt.imshow(images.cpu().numpy())
            plt.show()



            print(target)
            pred_logit_dict = model(image.float(), target.float())
            pred_logit = [value for value in pred_logit_dict.values() if value is not None]
            pred_logit = pred_logit[0]
            print(pred_logit.size()) # [num,lei] [4,500]
            indices = torch.max(pred_logit, dim=1)[1]
            print(indices)
            correct = torch.eq(indices, target).view(-1)
            print(correct)
            num_correct += torch.sum(correct).item()
            num_example += correct.shape[0]
            exit()
    acc = (num_correct / num_example)
    print("acc",acc)
    torch.cuda.empty_cache()


    end_time = datetime.datetime.now()
    time = (end_time - curr_time).total_seconds()
    if (time >= 3600):
        time_h = int(time / 3600)
        time_m = int(time % 3600 / 60)
        if (time_m >= 1):
            print("程序运行时长：", time_h, "h", time_m, "m", int(time % 60)+1, "s ")
        else:
            print("程序运行时长：", time_h, "h", int(time % 60)+1, "s ")
    else:
        time_m = int(time / 60)
        if (time_m >= 1):
            print("程序运行时长：", time_m, "m", int(time % 60)+1 ,"s ")
        else:
            print("程序运行时长：", int(time % 60)+1, "s ")

# python train.py -config_file "a" -tag "b" -max_num_devices 5
# F:\PycharmProjects_\Graduation\features\configs\base.py