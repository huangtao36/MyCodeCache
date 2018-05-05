# -*- coding:utf-8 -*-
from __future__ import print_function

import os
import numpy as np
from scipy import misc


# Converts a Tensor into a Numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 3:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    else:
        image_numpy = image_numpy[0] * 85.0

    image_numpy = np.abs(image_numpy)

    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    misc.imsave(image_path, image_numpy)  # N, M, 3


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
