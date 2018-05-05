import argparse
import os
from utilSet import util
import torch


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False
        self.opt = None
        self.isTrain = True

    def initialize(self):
        self.parser.add_argument('--dataroot', default='dataset')

        self.parser.add_argument('--step_size', type=int, default=19)

        self.parser.add_argument('--variable', default='save_result', type=str)
        self.parser.add_argument('--variable_value', default='first', type=str)

        self.parser.add_argument('--result_root_dir', default='checkpoints', type=str)

        self.parser.add_argument('--batchSize', type=int, default=1)

        self.parser.add_argument('--image_width', type=int, default=341)
        self.parser.add_argument('--image_height', type=int, default=256)
        self.parser.add_argument('--fineSize', type=int, default=256)

        self.parser.add_argument('--gpu_ids', type=str, default='0')

        self.parser.add_argument('--nThreads', default=2, type=int)

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

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

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

        self.isTrain = True

    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--epoch', type=int, default=200)
        self.parser.add_argument('--lr', type=float, default=0.0001)
        self.parser.add_argument('--sparse_k', type=int, default=10)

        self.parser.add_argument('--niter', type=int, default=100,
                                 help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100,
                                 help='# of iter to linearly decay learning rate to zero')

        self.parser.add_argument('--display_num', type=int, default=45)
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1)
        self.parser.add_argument('--phase', type=str, default='train')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')

        self.parser.add_argument('--no_lsgan', action='store_true',
                                 help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_A', type=float, default=10.0)
        self.parser.add_argument('--lambda_B', type=float, default=10.0)

        self.parser.add_argument('--no_html', action='store_true')

        self.parser.add_argument('--identity', type=float, default=0.5)


class TestOptions(BaseOptions):

    def __init__(self):
        super(TestOptions, self).__init__()
        self.isTrain = False

    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='test')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
