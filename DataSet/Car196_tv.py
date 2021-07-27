from __future__ import absolute_import, print_function
"""
CUB-200-2011 data-set for Pytorch
"""
import torch
import torch.utils.data as data
from PIL import Image

import os
from torchvision import transforms
from collections import defaultdict

from DataSet.CUB200 import MyData, default_loader, Generate_transform_Dict


class Cars196:
    def __init__(self, root=None, origin_width=256, width=227, ratio=0.16, transform=None):
        if transform is None:
            transform_Dict = Generate_transform_Dict(origin_width=origin_width, width=width, ratio=ratio)
        if root is None:
            root = 'data/Cars196/'

        train_txt = os.path.join(root, 'train.txt')
        test_txt = os.path.join(root, 'test.txt')
        #randomvalidset
        # val_v1_txt = os.path.join(root, 'val_v1.txt')
        # val_v1_7_txt = os.path.join(root, 'val_v1_7.txt')
        # val_v1_9_txt = os.path.join(root, 'val_v1_9.txt')
        # #hardvalidset
        # val_h_5_txt = os.path.join(root, 'val_set_h_5.txt')
        # val_h_7_txt = os.path.join(root, 'val_set_h_7.txt')
        # val_h_9_txt = os.path.join(root, 'val_set_h_9.txt')
        # val_v2_txt = os.path.join(root, 'val_v2.txt')
        # train_sub_val_v1_txt = os.path.join(root, 'train_sub_val_v1.txt')
        # train_sub_val_v2_txt = os.path.join(root, 'train_sub_val_v2.txt')

        self.train = MyData(root, label_txt=train_txt, transform=transform_Dict['rand-crop'])
        # self.val_v1 = MyData(root, label_txt=val_v1_txt, transform=transform_Dict['rand-crop'])
        # self.val_v1_7 = MyData(root, label_txt=val_v1_7_txt, transform=transform_Dict['rand-crop'])
        # self.val_v1_9 = MyData(root, label_txt=val_v1_9_txt, transform=transform_Dict['rand-crop'])
        # self.val_h_5 = MyData(root, label_txt=val_h_5_txt, transform=transform_Dict['rand-crop'])
        # self.val_h_7 = MyData(root, label_txt=val_h_7_txt, transform=transform_Dict['rand-crop'])
        # self.val_h_9 = MyData(root, label_txt=val_h_9_txt, transform=transform_Dict['rand-crop'])
        # self.val_v2 = MyData(root, label_txt=val_v2_txt, transform=transform_Dict['rand-crop'])
        # self.train_sub_val_v1 = MyData(root, label_txt=train_sub_val_v1_txt, transform=transform_Dict['rand-crop'])
        # self.train_sub_val_v2 = MyData(root, label_txt=train_sub_val_v2_txt, transform=transform_Dict['rand-crop'])
        self.gallery = MyData(root, label_txt=test_txt, transform=transform_Dict['center-crop'])


def testCar196():
    data = Cars196()
    print(len(data.gallery))
    print(len(data.train))
    print(data.train[1])


if __name__ == "__main__":
    testCar196()


