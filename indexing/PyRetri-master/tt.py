'''
@Author      : WangPeide
@Contact     : 377079627@qq.com
LastEditors: Please set LastEditors
@Description :
LastEditTime: 2022/4/14 11:07
'''

#使用pickle模块从文件中重构python对象

import pprint, pickle

pkl_file = open(r'F:\PycharmProjects_\Graduation\indexing\PyRetri-master\data_jsons\huawei_A_query.json', 'rb')

data1 = pickle.load(pkl_file)
pprint.pprint(data1)

# data2 = pickle.load(pkl_file)
# pprint.pprint(data2)

pkl_file.close()
