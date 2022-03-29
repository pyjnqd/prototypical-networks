import os
import sys
import glob
import csv
from functools import partial

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import ToTensor

from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose

import protonets
from protonets.data.base import convert_dict, CudaTransform, EpisodicBatchSampler, SequentialBatchSampler

MINIIMAGENET_DATA_DIR  = '/home/wuhao/data/mini_imagenet'

MINIIMAGENET_CACHE = { }

def load_image_path(key, out_field, d):
    d[out_field] = Image.open(d[key])
    return d

def convert_tensor(key, d):
    d[key] = 1.0 - torch.from_numpy(np.array(d[key], np.float32, copy=False)).transpose(0, 1).contiguous().view(3, d[key].size[0], d[key].size[1])
    return d

def rotate_image(key, rot, d):
    d[key] = d[key].rotate(rot)
    return d

def scale_image(key, height, width, d):
    d[key] = d[key].resize((height, width))
    return d

def load_class_images(d):
    if list(d['class'].keys())[0] not in MINIIMAGENET_CACHE:
        class_name, image_name_list = list(d['class'].keys())[0], list(d['class'].values())[0]

        image_dir = os.path.join(MINIIMAGENET_DATA_DIR, 'images')
        class_images = []
        for name in image_name_list:
            class_images.append(os.path.join(image_dir, name))
        class_images = sorted(class_images)

        if len(class_images) == 0:
            raise Exception("No images found for miniImagenet class {} at {}. Did you download miniImagenet?".format(d['class'], image_dir))

        image_ds = TransformDataset(ListDataset(class_images),
                                    compose([partial(convert_dict, 'file_name'),
                                             partial(load_image_path, 'file_name', 'data'),
                                             # partial(rotate_image, 'data', float(rot[3:])),
                                             partial(scale_image, 'data', 84, 84),
                                             partial(convert_tensor, 'data')]))

        loader = torch.utils.data.DataLoader(image_ds, batch_size=len(image_ds), shuffle=False)

        for sample in loader:
            MINIIMAGENET_CACHE[list(d['class'].keys())[0]] = sample['data']
            break # only need one sample because batch size equal to dataset length

    return { 'class': list(d['class'].keys())[0], 'data': MINIIMAGENET_CACHE[list(d['class'].keys())[0]] }

def extract_episode(n_support, n_query, d):
    # data: N x C x H x W
    n_examples = d['data'].size(0)

    if n_query == -1:
        n_query = n_examples - n_support

    example_inds = torch.randperm(n_examples)[:(n_support+n_query)]
    support_inds = example_inds[:n_support]
    query_inds = example_inds[n_support:]

    xs = d['data'][support_inds]
    xq = d['data'][query_inds]

    return {
        'class': d['class'],
        'xs': xs,
        'xq': xq
    }

def load(opt, splits):
    
    split_dir = os.path.join(MINIIMAGENET_DATA_DIR, 'splits', opt['data.split'])

    ret = { }
    for split in splits:
        if split in ['val', 'test'] and opt['data.test_way'] != 0:
            n_way = opt['data.test_way']
        else:
            n_way = opt['data.way']

        if split in ['val', 'test'] and opt['data.test_shot'] != 0:
            n_support = opt['data.test_shot']
        else:
            n_support = opt['data.shot']

        if split in ['val', 'test'] and opt['data.test_query'] != 0:
            n_query = opt['data.test_query']
        else:
            n_query = opt['data.query']

        if split in ['val', 'test']:
            n_episodes = opt['data.test_episodes']
        else:
            n_episodes = opt['data.train_episodes']

        transforms = [partial(convert_dict, 'class'),
                      load_class_images,
                      partial(extract_episode, n_support, n_query)]
        if opt['data.cuda']:
            transforms.append(CudaTransform())

        transforms = compose(transforms)

        class_names = []
        with open(os.path.join(split_dir, "{:s}.csv".format(split)), 'r') as f:
            f_csv = csv.reader(f)
            headers = next(f_csv)
            for row in f_csv:
                filename = row[0]
                label = row[1]
                if len(class_names) == 0:
                    dict_tmp = {label: [filename]}
                    class_names.append(dict_tmp)
                elif list(class_names[-1].keys())[0] == label:
                    list(class_names[-1].values())[0].append(filename)
                else:
                    dict_tmp = { label:[filename] }
                    class_names.append(dict_tmp)
        ds = TransformDataset(ListDataset(class_names), transforms)

        if opt['data.sequential']:
            sampler = SequentialBatchSampler(len(ds))
        else:
            sampler = EpisodicBatchSampler(len(ds), n_way, n_episodes)

        # use num_workers=0, otherwise may receive duplicate episodes
        ret[split] = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0)

    return ret
