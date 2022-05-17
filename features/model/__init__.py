'''
@Author      : WangPeide
@Contact     : 377079627@qq.com
LastEditors: Please set LastEditors
@Description :
LastEditTime: 2022/1/10 23:09
'''


# from .build import FullModel
import copy
import torch
# import net as Net
import sys

# this.py自用
sys.path.append('..')
from configs.base import config as cfg
# import net as Net

# pack_cmd 用
sys.path.append('..')
# from configs.base import config as cfg
from .import net as Net
# "F:\PycharmProjects_\Graduation\features\checkpoints\20220209_tag_example\tag_example_4.pth"
def build_model(cfg,pretrain_path=''):
    cfg = copy.deepcopy(cfg)
    if 'net' in cfg['model'].keys():
        net_cfg = cfg['model']['net']
        net_type = net_cfg.pop("type")
        model = getattr(Net,net_type)(cfg)
    else:
        raise KeyError("`net` not in cfg['model']")
    if pretrain_path:
        pass
        # print("perfect")
        # print(pretrain_path)

        model_state_dict = model.state_dict()
        state_dict = torch.load(pretrain_path,map_location='cpu')
        # print(state_dict.keys())  # dict_keys(['model', 'cfg'])
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        for key in state_dict.keys():
            # print(key)
            if key in model_state_dict.keys() and state_dict[key].shape==model_state_dict[key].shape:
                # print("perfect")
                model_state_dict[key] = state_dict[key]
        model.load_state_dict(model_state_dict)
    return model

if __name__ == "__main__":
    res = build_model(cfg)
    print(type(res))
    loss = res(inputs=torch.ones(5,3,512,512),targets=torch.arange(1, 6)) # torch.arange(1,6)  torch.ones(5)
    print(loss)
    # tensorboard - -logdir =./ features / log
    pass
