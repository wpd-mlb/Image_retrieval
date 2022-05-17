'''
@Author      : WangPeide
@Contact     : 377079627@qq.com
LastEditors: Please set LastEditors
@Description :
LastEditTime: 2022/1/2 15:51
'''

config = dict(
    # Basic Config
    enable_backends_cudnn_benchmark = True, # cudnn实现网络的加速
    max_epochs = 200+1,
    log_period = 0.05, # 一次打印 5%
    save_dir = r"../checkpoints/",
    log_dir = r"../log/",

    train_pipeline=dict(
        dataloader=dict(batch_size=2, num_workers=4, drop_last=True, pin_memory=False,
                        collate_fn="my_collate_fn"),

        dataset=dict(type="train_dataset",
                     root_dir=r"F:\PycharmProjects_\Graduation\features\data\train_data_resize512",
                     images_per_classes=2, classes_per_minibatch=1),

        transforms=[
            # dict(type="RescalePad",output_size=320),
            # dict(type="MultisizePad",p=0.1,resizes=[384,448,320],padsize=512),
            dict(type="ShiftScaleRotate", p=0.3, shift_limit=0.1, scale_limit=(-0.5, 0.2), rotate_limit=15),
            # dict(type="IAAPerspective", p=0.1, scale=(0.05, 0.15)),
            dict(type="ChannelShuffle", p=0.1),
            # dict(type="RandomCropResized",p=0.15,output_size=(512,512),scale=(0.85,0.95),ratio=(3/4,4/3)),
            dict(type="RandomRotate90", p=0.2),
            # dict(type="RandomRotation",degrees=(-15,15)),
            dict(type="RandomHorizontalFlip", p=0.5),
            dict(type="RandomVerticalFlip", p=0.5),
            dict(type="ColorJitter", brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            dict(type="RandomErasing", p=0.2, sl=0.02, sh=0.2, rl=0.2),
            dict(type="RandomPatch", p=0.05, pool_capacity=1000, min_sample_size=100, patch_min_area=0.01,
                 patch_max_area=0.2, patch_min_ratio=0.2, p_rotate=0.5, p_flip_left_right=0.5),
            dict(type="ToTensor", ),
            dict(type="Normalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True),
        ],

    ),
    # orignal batch_size 30
    gallery_pipeline = dict(
        dataloader = dict(batch_size = 1,shuffle = False,num_workers = 0,drop_last = False),
        dataset = dict(type="load_npy",
                    image_dir = r"F:\PycharmProjects_\Graduation\features\data\test_data_A_resize512\gallery",),
                                # r"F:\PycharmProjects_\Graduation\features\data\AA_numpy"
        transforms = [
            dict(type="ToTensor",),
            dict(type="Normalize",mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=False),
            ],

    ),
    query_pipeline = dict(
        dataloader = dict(batch_size = 1,shuffle = False,num_workers = 0,drop_last = False),
        dataset = dict(type="load_npy",
                    image_dir = r"F:\PycharmProjects_\Graduation\features\data\test_data_A_resize512\query",),
        transforms = [
            # dict(type="RescalePad",output_size=320),
            dict(type="ToTensor",),
            dict(type="Normalize",mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=False),
            ],
    ),
    # Model
    # model :
    ##      --> backbone : 特征提取，需在model/backbone中定义
    ##      --> aggregation : pooling_layer,model/aggregation
    ##      --> heads : classification heads,model/heads
    ##      --> losses: criterion. model/losses
    model=dict(
        net=dict(type="SBNet"),
        backbone=dict(type="densenet169", pretrained=False),
        aggregation=dict(type="GeneralizedMeanPoolingP", output_size=(1, 1), ),
        # in_feat is network output using backbone
        heads=dict(type="BNneckHead", in_feat=1664, num_classes=501), # densenet 1664 c fin 3097
        losses=[
            # dict(type="AMLinear",in_features=1664,num_clssses=3097,m=0.35,s=30,weight=1/4),
            dict(type="ArcfaceLoss_Dropout", in_feat=1664, num_classes=501, scale=35, margin=0.30, dropout_rate=0.2,
                 weight=1),
            dict(type="TripletLoss", margin=0.6, weight=1.0),
        ]
    ),

    multi_gpu=True,
    max_num_devices=1,  # 自动获取空闲显卡，默认第一个为主卡

    # Solver
    ## lr_scheduler : 学习率调整策略，默认从 torch.optim.lr_scheduler 中加载
    ## optimizer : 优化器，默认从 torch.optim 中加载
    # lr_scheduler = dict(type="ExponentialLR",gamma=0.9999503585), # cycle_momentum=False if optimizer==Adam
    lr_scheduler=dict(type="ExponentialLR", gamma=0.99998),  # cycle_momentum=False if optimizer==Adam

    optimizer=dict(type="Adam", lr=4e-4, weight_decay=1e-5),
    # optimizer = dict(type="AdamW",lr=2e-4),
    warm_up=dict(length=2000, min_lr=4e-6, max_lr=4e-4, froze_num_lyers=8)
    # Outpu
    # save_dir = r""

)

if __name__ == "__main__":
    pass
    for i in config['model'].keys():
        print(i)