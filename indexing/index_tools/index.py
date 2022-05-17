'''
@Author      : WangPeide
@Contact     : 377079627@qq.com
LastEditors: Please set LastEditors
@Description :
LastEditTime: 2022/4/5 12:01
'''

import argparse
import os
import pickle
import time
from os.path import join,dirname,realpath

from pyretri.config import get_defaults_cfg, setup_cfg
from pyretri.index import build_index_helper, feature_loader
from pyretri.evaluate import build_evaluate_helper

def save_result_n(save_file,index_result_info,query_info,gallery_info):
    fp = open(save_file, 'w', encoding='utf-8')
    num=len(index_result_info)

    for i in range(num):
        lst_tmp=[]
        img_name=query_info[i]['path'].split('/')[-1][:-4]+'.jpg'
        lst_tmp.append(img_name)
        res_tmp='{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,'.format\
            (gallery_info[index_result_info[i]['ranked_neighbors_idx'][0]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][1]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][2]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][3]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][4]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][5]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][6]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][7]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][8]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][9]]['path'].split('/')[-1][:-4])
        res_tmp+='{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,'.format\
            (gallery_info[index_result_info[i]['ranked_neighbors_idx'][10]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][11]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][12]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][13]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][14]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][15]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][16]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][17]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][18]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][19]]['path'].split('/')[-1][:-4])
        res_tmp+='{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg'.format\
            (gallery_info[index_result_info[i]['ranked_neighbors_idx'][20]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][21]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][22]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][23]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][24]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][25]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][26]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][27]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][28]]['path'].split('/')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][29]]['path'].split('/')[-1][:-4])
        res_tmp=img_name+',{'+res_tmp+'}'
        fp.write(res_tmp)
        fp.write('\n')

def save_result(save_file,index_result_info,query_info,gallery_info):
    fp = open(save_file, 'w', encoding='utf-8')
    num=len(index_result_info)
    print(gallery_info[0])
    for i in range(num):
        lst_tmp=[]

        # print(query_info[i])
        # {'path': 'F:\\PycharmProjects_\\Graduation\\features\\data\\test_data_A\\query\\01HMATBIX4CFKDS8.jpg',
        #  'label': '01HMATBIX4CFKDS8.jpg',
        #  'query_name': '01HMATBIX4CFKDS8.jpg',
        #  'idx': 0,
        #  'ranked_neighbors_idx':[]}
        img_name=query_info[i]['path'].split('\\')[-1][:-4]+'.jpg'
        lst_tmp.append(img_name)
        res_tmp='{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg,{}.jpg'.format\
            (gallery_info[index_result_info[i]['ranked_neighbors_idx'][0]]['path'].split('\\')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][1]]['path'].split('\\')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][2]]['path'].split('\\')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][3]]['path'].split('\\')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][4]]['path'].split('\\')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][5]]['path'].split('\\')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][6]]['path'].split('\\')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][7]]['path'].split('\\')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][8]]['path'].split('\\')[-1][:-4],\
            gallery_info[index_result_info[i]['ranked_neighbors_idx'][9]]['path'].split('\\')[-1][:-4])
        res_tmp=img_name+','+res_tmp+''
        fp.write(res_tmp)
        fp.write('\n')

#     fp.close()

def parse_args():
    parser = argparse.ArgumentParser(description='A tool box for deep learning-based image retrieval')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--config_file', '-cfg', default="../index_configs/hw_json.yaml", metavar='FILE', type=str, help='path to config file')
    parser.add_argument('--save_file', '-sf', default='../result/submission.csv', metavar='FILE', type=str, help='path to config file')
    args = parser.parse_args()
    return args

def main():
    start=time.time()

    # init args
    args = parse_args()
    assert args.config_file is not None, 'a config file must be provided!'

    # init and load retrieval pipeline settings
    cfg = get_defaults_cfg()
    cfg = setup_cfg(cfg, args.config_file, args.opts)

    # ****


    # load features
    query_fea, query_info, _ = feature_loader.load(cfg.index.query_fea_dir, cfg.index.feature_names)
    gallery_fea, gallery_info, _ = feature_loader.load(cfg.index.gallery_fea_dir, cfg.index.feature_names)

    print("using init_load time: {:6f}s".format(time.time() - start))
    # build helper and index features
    index_helper = build_index_helper(cfg.index)
    index_result_info, query_fea, gallery_fea = index_helper.do_index(query_fea, query_info, gallery_fea)

    # build helper and evaluate results
    evaluate_helper = build_evaluate_helper(cfg.evaluate)
    mAP, recall_at_k = evaluate_helper.do_eval(index_result_info, gallery_info)

    # show results
    evaluate_helper.show_results(mAP, recall_at_k)

    # save_result(args.save_file, index_result_info, query_info, gallery_info)
    # save_result_n(r"../result_tmp/submission_30.csv", index_result_info, query_info, gallery_info)

    print("using total time: {:6f}s".format(time.time() - start))
    # print(mAP)
    # print(recall_at_k)

if __name__ == '__main__':
    main()
# 结果
# [LoadFeature]: loading feature from F:/PycharmProjects_/Graduation/indexing/features/query\part_0.json...
# [LoadFeature] Success, total 500 images,
#  feature names: dict_keys(['f1'])
# [LoadFeature]: loading feature from F:/PycharmProjects_/Graduation/indexing/features/gallery\part_0.json...
# [LoadFeature] Success, total 1000 images,
#  feature names: dict_keys(['f1'])
# using init_load time: 0.390595s
# using total time: 0.546865s

# --------------- Retrieval Evaluation ------------
# mAP: 100.0
# R@1: 100.0	R@2: 100.0	R@4: 100.0	R@8: 100.0
# using total time: 0.046863s