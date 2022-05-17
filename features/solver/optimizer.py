'''
@Author      : WangPeide
@Contact     : 377079627@qq.com
LastEditors: Please set LastEditors
@Description :
LastEditTime: 2022/1/27 21:54
'''
from torch import optim


def make_optimizer(cfg_optimizer,model):
    cfg_optimizer = cfg_optimizer.copy()
    optimizer_type = cfg_optimizer.pop('type')
    if hasattr(optim,optimizer_type):
        params = model.parameters()
        optimizer = getattr(optim,optimizer_type)(params,**cfg_optimizer)
        return optimizer
    else:
        raise KeyError("optimizer not found. Got {}".format(optimizer_type))

