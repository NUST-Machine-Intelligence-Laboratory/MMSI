from __future__ import absolute_import, print_function

"""
CUB-200-2011 data-set for Pytorch
"""
import torch
import torch.utils.data as data
from PIL import Image
import torchvision
import os
import sys
from DataSet import transforms
from collections import defaultdict
import numpy as np

def default_loader(path):
    return Image.open(path).convert('RGB')

loader = torchvision.transforms.Compose([transforms.ToTensor()])
unloader = torchvision.transforms.ToPILImage()
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image
def PIL_to_tensor(image):
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)
def Generate_transform_Dict(origin_width=256, width=227, ratio=0.16):
    std_value = 1.0 / 255.0
    normalize = transforms.Normalize(mean=[104 / 255.0, 117 / 255.0, 128 / 255.0],
                                     std=[1.0 / 255, 1.0 / 255, 1.0 / 255])

    transform_dict = {}

    transform_dict['rand-crop'] = \
        transforms.Compose([
            #transforms.CovertBGR(),
            transforms.ToPILImage(),
            transforms.Resize((origin_width)),
            transforms.RandomResizedCrop(scale=(ratio, 1), size=width),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    transform_dict['center-crop'] = \
        transforms.Compose([
            #transforms.CovertBGR(),

            transforms.Resize((origin_width)),
            transforms.CenterCrop(width),
            transforms.ToTensor(),
            normalize,
        ])

    transform_dict['resize'] = \
        transforms.Compose([
            #transforms.CovertBGR(),
            transforms.Resize((width)),
            transforms.ToTensor(),
            normalize,
        ])
    return transform_dict


def Augment_Data(Inputs, Labels, transform=None, loader=default_loader, freq=1):
    # Initialization data path and train(gallery or query) txt path
    if transform is None:
        transform_dict = Generate_transform_Dict()['rand-crop']
    images = []
    #print("images:", Inputs.size())
    #print("labels:", Labels.size())
    num=Inputs.size(0)

    # Generate Index Dictionary for every class


    # Initialization Done
    re_label=[]
    re_img=[]
    for n in range(0,num):
        for f in range(0,freq):
            #print(num, "ori:", Inputs[num])
            pic=Inputs[n]
            if transform is not None:
                i = transform(pic)
                #i = PIL_to_tensor(i)

                re_img.append(i.numpy())
                #print(num, ":", f, ":", i)
                re_label.append(Labels[n].item())
    re_label=torch.LongTensor(re_label)
    re_img = torch.Tensor(re_img)
    print(re_img.size())
    print(re_label.size())
    return re_img, re_label




def Aug(data=None, width=227, origin_width=256, ratio=0.16, aug_instance=1):
    #print('width: \t {}'.format(width))
    inputs, labels = data
    #print("inputs:", inputs)
    #print("labels", labels)

    transform_Dict = Generate_transform_Dict(origin_width=origin_width, width=width, ratio=ratio)
    data_ = Augment_Data(inputs, labels, transform=transform_Dict['rand-crop'], freq=aug_instance)
    return data_

