# -*- coding:utf-8 -*-
import os
import os.path
import torch
from scipy import misc
import numpy as np
import torch.utils.data as data

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_image_files(dir_name):
    images = []
    assert os.path.isdir(dir_name), '%s is not a valid directory' % dir_name

    for root, _, f_names in sorted(os.walk(dir_name)):
        for f_name in f_names:
            if is_image_file(f_name):
                path = os.path.join(root, f_name)
                images.append(path)

    return images


def center_crop(a_img, b_img, c_img, opt):

    th, tw = opt.height, opt.width

    x1, y1 = 304, 148
    a_img = a_img[y1:(y1 + th), x1:(x1 + tw), :]
    b_img = b_img[y1:(y1 + th), x1:(x1 + tw)]
    c_img = c_img[y1:(y1 + th), x1:(x1 + tw)]

    return a_img, b_img, c_img


'''
def down_crop(a_img, b_img, c_img):

    a_img = a_img[120:, :, :]
    b_img = b_img[120:, :]
    c_img = c_img[120:, :]

    return a_img, b_img, c_img


def resize(a_img, b_img, c_img):

    a_img = misc.imresize(a_img, [176, 608], interp='nearest')
    b_img = misc.imresize(b_img, [176, 608], interp='nearest')
    c_img = misc.imresize(c_img, [176, 608], interp='nearest')

    return a_img, b_img, c_img


def spase_sampling(sparse):
    random_array = np.random.randint(0, 2, (176, 608))
    new_sparse = sparse * random_array

    # random_array1 = np.random.randint(0, 2, (176, 608))
    # new_sparse = new_sparse * random_array1

    return new_sparse
'''


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class Dataset(BaseDataset):

    def __init__(self, opt):
        super(Dataset, self).__init__()

        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join('dataset', opt.dataroot, opt.phase, 'rgb')
        self.dir_B = os.path.join('dataset', opt.dataroot, opt.phase, 'groundtruth')
        self.dir_C = os.path.join('dataset', opt.dataroot, opt.phase, 'sparse'+opt.sparse_per)

        self.A_paths = get_image_files(self.dir_A)
        self.B_paths = get_image_files(self.dir_B)
        self.C_paths = get_image_files(self.dir_C)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.C_size = len(self.C_paths)

    def __getitem__(self, index):
        a_path = self.A_paths[index % self.A_size]
        b_path = self.B_paths[index % self.B_size]
        c_path = self.C_paths[index % self.C_size]

        a_img = misc.imread(a_path)  # 352, 1216, 3
        b_img = misc.imread(b_path)  # 352, 1216
        c_img = misc.imread(c_path)  # 352, 1216

        a_img, b_img, c_img = center_crop(a_img, b_img, c_img, self.opt)

        a_img = a_img.transpose((2, 0, 1))
        b_img = np.expand_dims(b_img, axis=0)
        c_img = np.expand_dims(c_img, axis=0)

        # to tensor[0, 1]
        a = torch.from_numpy(a_img / 255.0).float()
        b = torch.from_numpy(b_img / 85.0).float()
        c = torch.from_numpy(c_img / 85.0).float()

        return {'A': a,
                'B': b,
                'C': c,
                'A_paths': a_path,
                'B_paths': b_path,
                'C_paths': c_path}

    def __len__(self):
        return max(self.A_size, self.B_size, self.C_size)

    def name(self):
        return 'UnalignedDataset'


class DataLoader:

    def __init__(self, opt):
        self.opt = opt
        self.dataset = Dataset(opt)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=opt.batch_size,
                                                      shuffle=False,
                                                      num_workers=2)

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data_list in enumerate(self.dataloader):

            yield data_list
