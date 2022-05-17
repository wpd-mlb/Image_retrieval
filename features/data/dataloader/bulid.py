'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-08-12 15:15:27
Description : 
'''
# this.py调用
# import sys
# sys.path.append('..')
# from dataset import build_dataset
# from transforms import build_transforms
# import sampler as Samplers
# import collate_fn as my_collate_fn
# from torch.utils.data import DataLoader

# pack_cmd 用
from ..dataset import build_dataset
from ..transforms import build_transforms
from .import sampler as Samplers
from .import collate_fn as my_collate_fn
from torch.utils.data import DataLoader

train_pipeline = dict(
    dataloader=dict(batch_size=2, num_workers=2, drop_last=True, pin_memory=False,
                    collate_fn="my_collate_fn"),

    dataset=dict(type="train_dataset",
                 root_dir=r"F:\PycharmProjects_\Graduation\features\data\train_data_resize512",
                 images_per_classes=4, classes_per_minibatch=1),

    transforms=[
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
    ],

)


def make_dataloader(cfg_data_pipeline):

    cfg_data_pipeline = cfg_data_pipeline.copy()
    cfg_dataset = cfg_data_pipeline.pop('dataset')
    cfg_transforms = cfg_data_pipeline.pop('transforms')
    cfg_dataloader = cfg_data_pipeline.pop('dataloader')

    # print(cfg_dataset)
    transforms = build_transforms(cfg_transforms)
    dataset = build_dataset(cfg_dataset,transforms)

    if 'sampler' in cfg_dataloader:
        cfg_sample = cfg_dataloader.pop('sampler')
        sample_type = cfg_sample.pop('type')
        sampler = getattr(Samplers,sample_type)(dataset.label,**cfg_sample)
        dataloader = DataLoader(dataset,sampler=sampler,**cfg_dataloader)
        return dataloader
    else:
        if "collate_fn" in cfg_dataloader:
            cfg_collate_fn = cfg_dataloader.pop("collate_fn")
            if hasattr(my_collate_fn,cfg_collate_fn):  # has attribute
                collate_fn = getattr(my_collate_fn,cfg_collate_fn)  # get attribute
                dataloader = DataLoader(dataset,collate_fn=collate_fn,**cfg_dataloader)
                return dataloader
        else:
            # print("bulid.py")
            dataloader = DataLoader(dataset,**cfg_dataloader)
            return dataloader
    # dataloader = DataLoader(dataset, batch_size=9, num_workers=2)


if __name__ == "__main__":
    pass
    # dataloader = dict(batch_size=7, num_workers=8, drop_last=True, pin_memory=False,
    #                   collate_fn="my_collate_fn")
    # dataset = build_dataset(train_pipeline["dataset"], train_pipeline["transforms"])
    # train_dataloader = DataLoader(dataset,batch_size=9, num_workers=2)
    train_dataloader = make_dataloader(train_pipeline)
    images, labels = next(iter(train_dataloader))  # [batch_size,channel,28,28] [batch_size,1]
    print("batch的shape：",images.shape, labels.shape)
    x = images[0,:,:,:]
    x = x.permute(1,2,0)  # add
    # image = x.permute(0, 3)  # add
    import matplotlib.pyplot as plt
    plt.imshow(x)
    plt.show()
    print(x.size())
    print(labels)
    # for i, (input_data, target) in enumerate(train_dataloader):
    #     print('input_data%d' % i, input_data)
    #     print('target%d' % i, target)
    # a,b = dataset.__getitem__(46)
    # print(b)