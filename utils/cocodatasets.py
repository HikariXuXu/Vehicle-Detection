import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from pycocotools.coco import COCO
from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    
    def __init__(self, ann_file='annotations/instances_train2017.json', data_root='data/coco', img_size=416, mode='train'):
        self.ann_file = ann_file
        self.data_root = data_root
        
        if self.data_root is not None:
            if not os.path.isabs(self.ann_file):
                self.ann_file = os.path.join(self.data_root, self.ann_file)
        
        self.data_infos = self.load_annotations(self.ann_file)
        if mode == 'train':
            self.files = [self.data_root + '/train2017/' + self.data_infos[i]['filename'] for i in range(len(self.data_infos))]
        elif mode == 'val':
            self.files = [self.data_root + '/val2017/' + self.data_infos[i]['filename'] for i in range(len(self.data_infos))]
        else:
            self.files = [self.data_root + '/test2017/' + self.data_infos[i]['filename'] for i in range(len(self.data_infos))]
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)
    
    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

class CocoDataset(Dataset):
    
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    def __init__(self, 
                 ann_file='annotations/instances_train2017.json', 
                 data_root='data/coco', 
                 img_size=416, 
                 mode='train', 
                 augment=True, 
                 multiscale=True, 
                 normalized_labels=False):
        self.ann_file = ann_file
        self.data_root = data_root
        
        if self.data_root is not None:
            if not os.path.isabs(self.ann_file):
                self.ann_file = os.path.join(self.data_root, self.ann_file)
        
        self.data_infos = self.load_annotations(self.ann_file)
        if mode == 'train':
            self.files = [self.data_root + '/train2017/' + self.data_infos[i]['filename'] for i in range(len(self.data_infos))]
        elif mode == 'val':
            self.files = [self.data_root + '/val2017/' + self.data_infos[i]['filename'] for i in range(len(self.data_infos))]
        else:
            self.files = [self.data_root + '/test2017/' + self.data_infos[i]['filename'] for i in range(len(self.data_infos))]
        
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.files[index]

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        targets = None
        coco_id_map={1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10,
                     13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20,
                     23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 
                     36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40,
                     47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50,
                     57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60,
                     70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70,
                     81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79}
        img_id = self.data_infos[index]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        cat_ids = [coco_id_map[ann['category_id']] for ann in ann_info]
        bboxes = [ann['bbox'] for ann in ann_info]
        
        cat_ids = torch.from_numpy(np.array(cat_ids))
        bboxes = torch.from_numpy(np.array(bboxes))
            
        if len(bboxes.size()) == 2:
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (bboxes[:, 0] - bboxes[:, 2] / 2)
            y1 = h_factor * (bboxes[:, 1] - bboxes[:, 3] / 2)
            x2 = w_factor * (bboxes[:, 0] + bboxes[:, 2] / 2)
            y2 = h_factor * (bboxes[:, 1] + bboxes[:, 3] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            bboxes[:, 0] = ((x1 + x2) / 2) / padded_w
            bboxes[:, 1] = ((y1 + y2) / 2) / padded_h
            bboxes[:, 2] *= w_factor / padded_w
            bboxes[:, 3] *= h_factor / padded_h

            targets = torch.zeros((len(bboxes), 6))
            targets[:, 1] = cat_ids
            targets[:, 2:] = bboxes

            # Apply augmentations
            if self.augment:
                if np.random.random() < 0.5:
                    img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.files)
    
    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos