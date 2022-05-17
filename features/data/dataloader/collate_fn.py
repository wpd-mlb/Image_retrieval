'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-08-12 15:09:46
'''
import torch


def my_collate_fn(batch_list):
    image_list = list()
    label_list = list()
    for i in range(len(batch_list)):
        image,labels = batch_list[i]
        # image = image.permute(0,3,1,2) # add
        image_list.append(image.squeeze(0))
        label_list.append(labels.squeeze(0))
    return torch.cat(image_list,dim=0),torch.cat(label_list,dim=0)
    
def concat(batch_list):
    image_list = list()
    label_list = list()
    for i in range(len(batch_list)):
        image,labels = batch_list[i]
        image_list.append(image.squeeze(0))
        label_list.append(labels.squeeze(0))
    # import pdb; pdb.set_trace()
    return torch.cat(image_list,dim=0),torch.cat(label_list,dim=0)

if __name__ == "__main__":
    pass
    x = torch.randn(4, 512, 512, 3)
    x = x.permute(0,3,1,2)
    print(x.size())