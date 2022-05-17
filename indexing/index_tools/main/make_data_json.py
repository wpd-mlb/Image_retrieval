# -*- coding: utf-8 -*-

import argparse
import pickle
import os
from pyretri.extract import make_data_json


def parse_args():
    parser = argparse.ArgumentParser(description='A tool box for deep learning-based image retrieval')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--dataset', '-d', default=r"F:\PycharmProjects_\Graduation\features\data\AA", type=str, help="path for the dataset that make the json file")
    parser.add_argument('--save_path', '-sp', default="../data_jsons/caltech_gallery.json", type=str, help="save path for the json file")
    parser.add_argument('--type', '-t', default=None, type=str, help="mode of the dataset")
    parser.add_argument('--ground_truth', '-gt', default=None, type=str, help="ground truth of the dataset")

    args = parser.parse_args()

    return args


def main():

    # init args
    args = parse_args()
    assert args.dataset is not None, 'the data must be provided!'
    assert args.save_path is not None, 'the save path must be provided!'
    # assert args.type is not None, 'the type must be provided!'

    # make data json
    # make_data_json(args.dataset, args.save_path, args.type, args.ground_truth)
    dataset_path=args.dataset
    save_path=args.save_path

    """
    Generate data json file for dataset collecting images with the same label one directory. e.g. CUB-200-2011.

    Args:
        dataset_path (str): the path of the dataset.
        save_ds_path (str): the path for saving the data json files.
    """
    info_dicts = list()
    img_dirs = os.listdir(dataset_path)
    label_list = list()
    label_to_idx = dict()
    for dir in img_dirs:
        print(dir)
        for root, _, files in os.walk(os.path.join(dataset_path, dir)):
            for file in files:
                info_dict = dict()
                info_dict['path'] = os.path.join(root, file)
                if dir not in label_list:
                    label_to_idx[dir] = len(label_list)
                    label_list.append(dir)
                info_dict['label'] = dir
                info_dict['label_idx'] = label_to_idx[dir]
                print(label_to_idx[dir])
                info_dicts += [info_dict]
    with open(save_path, 'wb') as f:
        pickle.dump({'nr_class': len(img_dirs), 'path_type': 'absolute_path', 'info_dicts': info_dicts}, f)


    print('make data json have done!')


if __name__ == '__main__':
    main()

# python main/make_data_json.py -d F:\PycharmProjects_\Graduation\indexing\index_tools\data_jsons\gallery -sp data_jsons/caltech_gallery.json -t general
# python main/make_data_json.py -d F:\PycharmProjects_\Graduation\indexing\index_tools\data_jsons\query -sp data_jsons/caltech_query.json -t general
