'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors : now more
LastEditTime: 2020-07-21 22:55:06
Description : 
'''
# this py
# import opencv_transforms as transforms

from . import opencv_transforms as transforms

ss = [
    # dict(type="RescalePad",output_size=320),
    dict(type="ShiftScaleRotate", p=0.3, shift_limit=0.1, scale_limit=(-0.5, 0.2), rotate_limit=15),
    # dict(type="IAAPerspective", p=0.1, scale=(0.05, 0.15)),
    dict(type="ChannelShuffle", p=0.1),
    dict(type="RandomRotate90", p=0.2),
    dict(type="RandomHorizontalFlip", p=0.5),
    dict(type="RandomVerticalFlip", p=0.5),
    dict(type="ColorJitter", brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    dict(type="RandomErasing", p=0.2, sl=0.02, sh=0.2, rl=0.2),
    dict(type="RandomPatch", p=0.05, pool_capacity=1000, min_sample_size=100, patch_min_area=0.01,
         patch_max_area=0.2, patch_min_ratio=0.2, p_rotate=0.5, p_flip_left_right=0.5),
    dict(type="ToTensor", ),
    dict(type="Normalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True),
]

def build_transforms(cfg_transforms):
    cfg_transforms = cfg_transforms.copy()
    transforms_list = list()
    for item in cfg_transforms:
        transforms_type = item.pop("type")
        transforms_kwags = item
        if hasattr(transforms,transforms_type):  # has attribute
            transforms_list.append(getattr(transforms,transforms_type)(**transforms_kwags))
            # print()
        else:
            # raise ValueError("\'type\' of transforms is not defined. Got {}".format(transforms_type))
            pass
    # print("transforms_list",transforms_list)
    return transforms.Compose(transforms_list)

if __name__ == "__main__":
    pass
    sss = build_transforms(ss)