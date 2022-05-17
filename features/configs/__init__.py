'''
@Author      : WangPeide
@Contact     : 377079627@qq.com
LastEditors: Please set LastEditors
@Description :
LastEditTime: 2022/1/2 15:38
'''

from argparse import ArgumentParser


def load_arg():
    parser = ArgumentParser(description="Pytorch Training")
    # default="F:/PycharmProjects_/Graduation/features/configs/base.py",
    parser.add_argument("-config_file", "--CONFIG_FILE", type=str, default="../configs/base.py", required=False, help="Path to config file")
    parser.add_argument("-tag", "--TAG", default='tag_example', type=str)
    parser.add_argument("-max_num_devices", default=1, type=int) # cfg中有

    # DATA
    parser.add_argument("-train_num_workers", "--DATA.DATALOADER.TRAIN_NUM_WORKERS", type=int,
                        help='Num of data loading threads. ')

    # MODEL
    # default = 'F:\\PycharmProjects_\\Graduation\\features\\checkpoints\\20220211_tag_example\\tag_example_130.pth',
    parser.add_argument("-load_path", type=str,default = 'F:/PycharmProjects_/Graduation/features/checkpoints/20220211_tag_example\\tag_example_130.pth',
                        help='Path of pretrained model. ')
    parser.add_argument("-device", type=int, nargs='+',default=[0],
                        help="list of device_id, e.g. [0,1,2]")

    # SOLVER
    # parser.add_argument("-max_epochs","--SOLVER.MAX_EPOCHS",type=int,
    #                     help="num of epochs to train (default:50)")
    # parser.add_argument('-optimizer',"--SOLVER.OPTIMIZER_NAME",type=str,
    #                     help="optimizer (default:SGD)")
    # parser.add_argument("-criterion","--SOLVER.CRITERION",type=str,
    #                     help="Loss Function ")
    # parser.add_argument("-lr","--SOLVER.LEARNING_RATE",type=float,
    #                     help="Learning rate ")
    # parser.add_argument("-lr_scheduler","--SOLVER.LR_SCHEDULER",type=str)

    # UTILS
    # parser.add_argument("-find_lr",action="store_true")

    arg = parser.parse_args()
    return arg


def merage_from_arg(config, arg):  # --> dict{},dict{}
    if arg['TAG']:
        config['tag'] = arg['TAG']
    else:
        config['tag'] = (((arg['CONFIG_FILE']).split('/')[-1]).split('.'))[0]
    # print("TAG : ", config['tag'])

    if arg['max_num_devices']:
        config['max_num_devices'] = arg['max_num_devices']

    return config