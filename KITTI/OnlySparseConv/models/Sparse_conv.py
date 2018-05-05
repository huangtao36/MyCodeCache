import torch
from torch.autograd import Variable
import torch.nn as nn
from collections import namedtuple
# import os
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() \
    else torch.FloatTensor

Opts = namedtuple("Opts", ["EPS"])
opts = Opts(0.0001)


def torch_scalar(f):

    if type(f) is float:
        return FloatTensor([f, ])
    else:
        raise TypeError("Type {} not supported".format(str(type(f))))


TS_ONE = torch_scalar(1.0)
TS_EPS = torch_scalar(opts.EPS)
TS_ZERO = torch_scalar(0.0)


class AllOneConv2d(nn.Conv2d):

    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False):
        super(AllOneConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, bias=bias)

        self.weight.data[...] = 1.0


def get_all_one_conv(kernel_size, stride, padding):

    c = AllOneConv2d(kernel_size=kernel_size, stride=stride, padding=padding)
    return c


class SparseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(SparseConv, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=False)
        self.mask_conv = AllOneConv2d(in_channels=1, out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding)
        self.b = nn.Parameter(FloatTensor([0.0]))

        self.pool = nn.MaxPool2d(kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding)

    def forward(self, feature, mask):
        mask_pool = mask
        masked_feature = feature * mask
        hfeature = self.conv1(masked_feature)

        mask_c = self.mask_conv(mask)
        mask_invc = Variable(TS_ONE) / torch.max(mask_c, Variable(TS_EPS))
        mask_invc.detach_()

        hout = hfeature * mask_invc + self.b

        mask_out = self.pool(mask_pool)
        return hout, mask_out
