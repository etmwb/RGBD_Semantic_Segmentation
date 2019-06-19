import os
import sys
import numpy as np
import random
import math
import cv2
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.utils.data as data
import torchvision.transforms as transform
import re
from tqdm import tqdm
from .base import BaseDataset

class Nyuv2Segmentation(BaseDataset):
    BASE_DIR = 'nyuv2'
    NUM_CLASS = 40
    def __init__(self, root=os.path.expanduser('~/zzyai/data'), split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(Nyuv2Segmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        # assert exists
        root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(root), "Please download the dataset!!"

        self.images, self.masks, self.depths, self.HHAs = _get_nyuv2_pairs(root, split)
        if split != 'vis':
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        if self.mode == 'vis':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        
        mask = Image.open(self.masks[index])
        depth = Image.open(self.depths[index])
        HHA = Image.open(self.HHAs[index])
        
        # synchrosized transform
        if self.mode == 'train':
            img, mask, depth, HHA = self._refineNet_transform(img, mask, depth, HHA)
        elif self.mode == 'val':
            img, mask, depth, HHA = self._val_sync_transform(img, mask, depth, HHA)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask, depth, HHA#, self.images[index].split('/')[-1]

    def _depthaware_transform(self, img, mask, depth, HHA):
        r"""Data augmentation used in Depth-aware CNN for RGBD Semantic Segmentation: https://arxiv.org/abs/1803.06791 
        Random flip, random scale+crop and color jitter are used.

        Arguments for data augmentation:
            Random_flip: 0.5.
            Random_scale: 0.76~1.75 
            Random_crop: 421, 321 when batch_size > 1 
                         0.6~0.9  when batch_size == 1
            colorjitter: 
        """

        width, height = mask.size
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            HHA = HHA.transpose(Image.FLIP_LEFT_RIGHT)
        
        scale = random.uniform(0.76, 1.75)
        oh = height * scale
        ow = (oh * width // height)
        oh = int(round(oh / 8) * 8)
        ow = int(round(ow / 8) * 8)

        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        depth = depth.resize((ow, oh), Image.BILINEAR)
        HHA = HHA.resize((ow, oh), Image.BILINEAR)
        
        # random crop size(deprecated)
        cropscale = random.uniform(0.6, 0.9)
        cropsizeh = int(oh * cropscale)
        cropsizew = int(ow * cropscale)

        cropsizeh = 321
        cropsizew = 421
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - cropsizew)
        y1 = random.randint(0, h - cropsizeh)
        img = img.crop((x1, y1, x1+cropsizew, y1+cropsizeh))
        mask = mask.crop((x1, y1, x1+cropsizew, y1+cropsizeh))
        depth = depth.crop((x1, y1, x1+cropsizew, y1+cropsizeh)) 
        HHA = HHA.crop((x1, y1, x1+cropsizew, y1+cropsizeh))
        # convert RGB to HSV is instablize in Image
        img = np.array(img)
        if random.random() > 0.1: 
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = hsv[:, :, 0] + np.random.rand() * 70 - 35
            hsv[:, :, 1] = hsv[:, :, 1] + np.random.rand() * 0.3 - 0.15
            hsv[:, :, 2] = hsv[:, :, 2] + np.random.rand() * 50 - 25
            hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 360.)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1.)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255.)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # final transform
        return self._tensor_transform(img, mask, depth, HHA)

    def _refineNet_transform(self, img, mask, depth, HHA):
        r"""Data augmentation used in Light-Weight RefineNet for Real-Time Semantic Segmentation: https://arxiv.org/abs/1810.03272
        Random flip, random scale and crop are used.

        Arguments for data augmentation:
            Random_flip: 0.5.
            Random_scale+pad: 0.5~2.0  
            Random_crop: 500, 500 
        """
        img, mask, depth, HHA = map(np.array, (img,mask,depth,HHA))
        height, width = mask.shape 
        shorted_side, crop_size = 350, 500 
        low_scale, high_scale = 0.5, 2.0 
        img_val = [123.675, 116.28 , 103.53]
        HHA_val = [132.431, 94.076, 118.477]
        msk_val = 255
        # resize shorter scale 
        min_side = min(height, width)
        scale = np.random.uniform(low_scale, high_scale)
        if min_side * scale < shorted_side: 
            scale = shorted_side * 1. / min_side 
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        depth = depth.astype(np.float32)
        depth = cv2.resize(depth, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        HHA = cv2.resize(HHA, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        #pad 
        height, width = mask.shape 
        h_pad = int(np.clip(((crop_size - height) + 1)//2, 0, 1e6))
        w_pad = int(np.clip(((crop_size - width) + 1)//2, 0, 1e6))
        pad = ((h_pad, h_pad), (w_pad, w_pad))
        img = np.stack([np.pad(img[:,:,c], pad,
                         mode='constant',
                         constant_values=img_val[c]) for c in range(3)], axis=2)
        mask = np.pad(mask, pad, mode='constant', constant_values=msk_val)
        depth = np.pad(depth, pad, mode='constant', constant_values=0)
        HHA = np.stack([np.pad(HHA[:,:,c], pad,
                         mode='constant',
                         constant_values=HHA_val[c]) for c in range(3)], axis=2)
        # random mirror
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
            depth = cv2.flip(depth, 1)
            HHA = cv2.flip(HHA, 1)
        # random rotate -10~10, mask using NN rotate
        # deg = random.uniform(-10, 10)
        # img = img.rotate(deg, resample=Image.BILINEAR)
        # mask = mask.rotate(deg, resample=Image.NEAREST)
        # pad crop
        # random crop crop_size
        height, width = mask.shape
        new_h = min(height, crop_size)
        new_w = min(width, crop_size)
        top = np.random.randint(0, height - new_h + 1)
        left = np.random.randint(0, width - new_w + 1)
        img = img[top: top + new_h,
                        left: left + new_w]
        mask = mask[top: top + new_h,
                    left: left + new_w]
        depth = depth[top: top + new_h,
                        left: left + new_w]
        HHA = HHA[top: top + new_h,
                    left: left + new_w]
        # final transform
        return self._tensor_transform(img, mask, depth, HHA)

    def _val_sync_transform(self, img, mask, depth, HHA):
        # final transform
        return self._tensor_transform(img, mask, depth, HHA)

    def _tensor_transform(self, img, mask, depth, HHA): 
        img, mask, depth, HHA = map(np.array, (img, mask, depth, HHA))

        img = img - np.asarray([122.675,116.669,104.008]) 
        img = img.transpose((2, 0, 1))[::-1, :, :].astype(np.float32)
        img = torch.from_numpy(img).float()

        mask = mask.astype(np.uint8)
        mask = torch.from_numpy(mask).long()

        depth = depth.astype(np.float32)/40. # /40 defined alpha of the depth similarity term in Depth-aware CNN
        depth = torch.from_numpy(depth).float() 

        HHA = HHA - np.asarray([132.431, 94.076, 118.477])
        HHA = HHA.transpose((2, 0, 1)).astype(np.float32)
        HHA = torch.from_numpy(HHA).float()

        return img, mask, depth, HHA  

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def _get_nyuv2_pairs(folder, split='train'):
    r"""
    nyuv2_training.lst: contain image path for all training images
    nyuv2_testing.lst:  contain image path for all testing images
    """

    def get_path_pairs(folder, split_f):
        img_paths = []
        mask_paths = []
        depth_paths = []
        HHA_paths = []
        with open(split_f, 'r') as paths:
            for line in tqdm(paths):
                imgpath, maskpath, depthpath, HHApath = line.strip().split(' ')
                if os.path.isfile(maskpath): 
                    img_paths += [imgpath]
                    mask_paths += [maskpath]
                    depth_paths += [depthpath]
                    HHA_paths += [HHApath]
                else:
                    print('cannot find the mask:', maskpath)
        return img_paths, mask_paths, depth_paths, HHA_paths

    if split == 'train':
        split_f = os.path.join(folder, 'nyuv2_training.lst')
        assert os.path.exists(split_f), "Please create .lst file for nyuv2 training dataset!!"
        img_paths, mask_paths, dpt_paths, HHA_paths = get_path_pairs(folder, split_f)
    elif split == 'val':
        split_f = os.path.join(folder, 'nyuv2_testing.lst')
        assert os.path.exists(split_f), "Please create .lst file for nyuv2 testing dataset!!"
        img_paths, mask_paths, dpt_paths, HHA_paths = get_path_pairs(folder, split_f)

    return img_paths, mask_paths, dpt_paths, HHA_paths