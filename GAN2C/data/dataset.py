# -*- coding:utf-8 -*-
import os
import os.path
import torch
import torchvision.transforms as transforms
from PIL import Image
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

    for root, _, fnames in sorted(os.walk(dir_name)):
        for f_name in fnames:
            if is_image_file(f_name):
                path = os.path.join(root, f_name)
                images.append(path)

    return images


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


def get_transform(opt):
    transform_list = []
    transform_list += [transforms.Scale((opt.image_width, opt.image_height), Image.ANTIALIAS),
                       transforms.CenterCrop(opt.fineSize),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class Dataset(BaseDataset):

    def __init__(self, opt):
        super(Dataset, self).__init__()

        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.a_paths = get_image_files(self.dir_A)
        self.b_paths = get_image_files(self.dir_B)

        self.a_paths = sorted(self.a_paths)
        self.b_paths = sorted(self.b_paths)
        self.A_size = len(self.a_paths)
        self.B_size = len(self.b_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):

        index_a = index % self.A_size
        a_path = self.a_paths[index_a % self.A_size]

        index_b = index % self.B_size
        b_path = self.b_paths[index_b % self.B_size]

        a_img = Image.open(a_path).convert('RGB')
        b_img = Image.open(b_path).convert('RGB')

        a = self.transform(a_img)
        b = self.transform(b_img)

        return {'A': a, 
                'B': b,
                'A_paths': a_path,
                'B_paths': b_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'Dataset'


class DataLoader:

    def __init__(self, opt):
        self.opt = opt
        self.dataset = Dataset(opt)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=2)

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data_list in enumerate(self.dataloader):
            yield data_list
