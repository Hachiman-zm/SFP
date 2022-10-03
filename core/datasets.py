# Data loading based on https://github.com/NVIDIA/flownet2-pytorch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms

import math
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor
from PIL import Image


class DAVIS(data.Dataset):
    def __init__(self,
                 # root='/home/hddd/hzm/Dataset/DAVIS2017/DAVIS-2017-trainval-480p/DAVIS/Annotations/480p',
                 root='/home/hddd/hzm/Dataset/DAVIS2017/DAVIS-2017-trainval-480p/DAVIS/Annotations/480p_single',
                 split='train'):
        self.category_dir = os.listdir(root)
        self.videos_dir = [os.path.join(root, x) for x in self.category_dir]
        self.pairs_list = []
        self.is_valid = split == 'valid'
        self.threshlod = 500
        self.train_file = r'/home/hddd/hzm/Dataset/DAVIS2017/DAVIS-2017-trainval-480p/DAVIS/ImageSets/2017/train.txt'
        self.valid_file = r'/home/hddd/hzm/Dataset/DAVIS2017/DAVIS-2017-trainval-480p/DAVIS/ImageSets/2017/val.txt'
        with open(self.train_file, 'r') as f:
            self.train_list = [x[:-1] for x in f.readlines()]
        with open(self.valid_file, 'r') as f:
            self.valid_list = [x[:-1] for x in f.readlines()]

        if split == 'train':
            for video in self.videos_dir:
                if video.split('/')[-1] not in self.train_list:
                    continue

                object_ids = os.listdir(video)
                for object_id in object_ids:
                    object_dir = osp.join(video, object_id)
                    images = sorted(glob(osp.join(object_dir, '*.png')))
                    first_frame = images[:-1]
                    seconde_frame = images[1:]
                    for item in zip(first_frame, seconde_frame):
                        # img1 = Image.open(self.pairs_list[item][0])
                        img1_id = int(item[0].split('/')[-1].split('.')[0])
                        img2_id = int(item[1].split('/')[-1].split('.')[0])
                        if img2_id == img1_id + 1:
                            self.pairs_list.append(item)

            self.transform = transforms.Compose([transforms.Resize([448, 832]), transforms.ToTensor()])
            print("{} pairs of frames for training...".format(len(self.pairs_list)))

        if split == 'valid':
            self.videos_dir = sorted(self.videos_dir)
            for video in self.videos_dir:
                if video.split('/')[-1] not in self.valid_list:
                    continue

                object_ids = os.listdir(video)
                for object_id in object_ids:
                    object_dir = osp.join(video, object_id)
                    images = sorted(glob(osp.join(object_dir, '*.png')))
                    first_frame = images[:-1]
                    seconde_frame = images[1:]
                    for item in zip(first_frame, seconde_frame):
                        # img1 = Image.open(self.pairs_list[item][0])
                        img1_id = int(item[0].split('/')[-1].split('.')[0])
                        img2_id = int(item[1].split('/')[-1].split('.')[0])
                        if img2_id == img1_id + 1:
                            self.pairs_list.append(item)

            self.transform = transforms.Compose([transforms.Resize([448, 832]), transforms.ToTensor()])
            print("{} pairs of frames for validation...".format(len(self.pairs_list)))

    def count_pixels(x):
        counts = np.arange(256)
        counts = np.zeros_like(counts)
        for i in x.ravel():
            counts[i] += 1
        return counts

    def __getitem__(self, item):
        item = item % len(self.pairs_list)
        img1 = Image.open(self.pairs_list[item][0])
        img2 = Image.open(self.pairs_list[item][1])
        img1 = (np.array(img1) * 1. / np.max(img1) * 255.).astype(np.uint8)
        img2 = (np.array(img2) * 1. / np.max(img2) * 255.).astype(np.uint8)
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        return self.transform(img1), self.transform(img2)

    def __len__(self):
        return len(self.pairs_list)


if __name__ == '__main__':
    dataloader = DAVIS(split='valid')

    img_torch = dataloader[0][0]
    img_np = (img_torch.numpy() * 255.).astype(np.uint8)
    np.set_printoptions(threshold=np.inf)
