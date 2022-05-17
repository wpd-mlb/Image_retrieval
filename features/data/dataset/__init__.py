'''
@Author      : wpd
@Contact     : 377079627@qq.com
LastEditors: Please set LastEditors
@Description :
LastEditTime: 2021/12/30 18:15
'''
# this.py自用
# import build as datasets

# cmd 用
from .import build as datasets


# a = dict(type="train_dataset",
#                root_dir=r"F:\PycharmProjects_\Graduation\features\data\train_data_resize512",
#                images_per_classes=4, classes_per_minibatch=1)
#
# dataloader = dict(batch_size=7, num_workers=8, drop_last=True, pin_memory=False,
#                   collate_fn="my_collate_fn")
#
# transforms = [
#     # dict(type="RescalePad",output_size=320),
#     dict(type="ShiftScaleRotate", p=0.3, shift_limit=0.1, scale_limit=(-0.5, 0.2), rotate_limit=15),
#     dict(type="IAAPerspective", p=0.1, scale=(0.05, 0.15)),
#     dict(type="ChannelShuffle", p=0.1),
#     dict(type="RandomRotate90", p=0.2),
#     dict(type="RandomHorizontalFlip", p=0.5),
#     dict(type="RandomVerticalFlip", p=0.5),
#     dict(type="ColorJitter", brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#     dict(type="RandomErasing", p=0.2, sl=0.02, sh=0.2, rl=0.2),
#     dict(type="RandomPatch", p=0.05, pool_capacity=1000, min_sample_size=100, patch_min_area=0.01, patch_max_area=0.2,
#          patch_min_ratio=0.2, p_rotate=0.5, p_flip_left_right=0.5),
#     dict(type="ToTensor", ),
#     dict(type="Normalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True),
# ]


def build_dataset(cfg_dataset, transforms):
    cfg_dataset = cfg_dataset.copy()
    dataset_type = cfg_dataset.pop("type")
    dataset_kwags = cfg_dataset

    if hasattr(datasets, dataset_type):  # has attribute
        dataset = getattr(datasets, dataset_type)(**dataset_kwags, transforms=transforms)
        # *list *tuple 直接将元素直接取出  **dict元素取出
        # ** 多参数的传入或变量的拆解
    else:
        raise ValueError("\'type\' of dataset is not defined. Got {}".format(dataset_type))
    return dataset

if __name__ == '__main__':
    pass
    # print(type(dataset))

    # build_dataset(a,a)
