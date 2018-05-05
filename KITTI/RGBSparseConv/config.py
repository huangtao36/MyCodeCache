# -*- coding:utf-8 -*-
import argparse
import torch


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False
        self.opt = None
        self.isTrain = True

    def initialize(self):
        self.parser.add_argument('--dataroot', default='dataset169')
        self.parser.add_argument('--sparse_per', default='1', type=str)

        self.parser.add_argument('--width', default='608', type=int)
        self.parser.add_argument('--height', default='176', type=int)

        self.parser.add_argument('--variable', default='Sparse_sample')
        self.parser.add_argument('--variable_value', default='1', type=str)

        self.parser.add_argument('--gpu_ids', type=str, default='0')

        self.parser.add_argument('--result_root_dir', default='result')

        self.parser.add_argument('--nThreads', default=2, type=int,
                                 help='# threads for loading data')
        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--no_dropout', action='store_true',
                                 help='no dropout for the generator')
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--init_type', type=str, default='xavier',
                                 help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--batch_size', type=int, default=1,
                                 help='input batch size')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain

        gpu_ids = str(self.opt.gpu_ids)
        str_ids = gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id_num = int(str_id)
            if id_num >= 0:
                self.opt.gpu_ids.append(id_num)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        return self.opt


class TrainOptions(BaseOptions):

    def __init__(self):
        super(TrainOptions, self).__init__()
        self.isTrain = None

    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--epoch', type=int, default=300)
        self.parser.add_argument('--lr', type=float, default=0.0002)
        self.parser.add_argument('--sparse_k', type=int, default=300)
        self.parser.add_argument('--fake_b_loss_k', type=int, default=10)

        self.parser.add_argument('--display_num', type=int, default=3)

        self.parser.add_argument('--lambda_A', type=float, default=10.0,
                                 help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0,
                                 help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--identity', type=float, default=0)

        self.parser.add_argument('--niter', type=int, default=100,
                                 help='of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=200,
                                 help='of iter to linearly decay learning rate to zero')

        self.parser.add_argument('--pool_size', type=int, default=50,
                                 help='the size of image buffer that stores previously generated images')

        self.parser.add_argument('--phase', type=str, default='train',
                                 help='train, val, test, etc')

        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1)
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--no_lsgan', action='store_true',
                                 help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--no_html', action='store_true')

        self.isTrain = True


class TestOptions(BaseOptions):

    def __init__(self):
        super(TestOptions, self).__init__()
        self.isTrain = None

    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='test',
                                 help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')

        self.isTrain = False
