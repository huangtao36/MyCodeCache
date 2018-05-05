# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
# from .Sparse_conv import SparseConv
from config import TrainOptions

opt = TrainOptions().parse()

# ----------------------------------------------------------------------------------
# import torch
from torch.autograd import Variable
# import torch.nn as nn
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
        mask_invc = Variable(TS_ONE) / torch.max(mask_c, Variable(TS_EPS)).cuda()
        mask_invc.detach_()

        hout = hfeature * mask_invc + self.b

        mask_out = self.pool(mask_pool)
        return hout, mask_out

# ----------------------------------------------------------------------------------


def weights_init_xavier(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1 and \
       class_name not in ['AllOneConv2d', 'SparseConv']:
        init.xavier_normal(m.weight.data)
    elif class_name.find('Linear') != -1:
        init.xavier_normal(m.weight.data)
    elif class_name.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


# 权重初始化
def init_weights(net, init_type='normal'):
    if init_type == 'xavier':
        net.apply(weights_init_xavier)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


# 归一化
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


# 获得学习率
def get_scheduler(optimizer):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler


# 生成器
def define_g_a(input_nc, input_sparse_nc, output_nc, norm='instance', init_type='xavier', gpu_ids=[]):

    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    net_g_a = GenerateA(input_nc=input_nc, input_sparse_nc=input_sparse_nc,
                        output_nc=output_nc, norm_layer=norm_layer, gpu_ids=gpu_ids)

    if len(gpu_ids) > 0:
        net_g_a.cuda(gpu_ids[0])

    init_weights(net_g_a, init_type=init_type)

    return net_g_a


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def _full_conv(conv_op, input_a, input_c, mask):

    multi = input_c * mask.float()
    combined_signal = torch.cat([input_a, multi], dim=1)
    return conv_op(combined_signal)


class GenerateA(nn.Module):
    def __init__(self, input_nc=3, input_sparse_nc=1, output_nc=1,
                 n_base_filters=64,
                 n_sparse_filters=16,
                 norm_layer=nn.BatchNorm2d,
                 n_blocks=9, gpu_ids=[]):

        assert (n_blocks >= 0)
        super(GenerateA, self).__init__()

        self.input_nc = input_nc
        self.n_base_filters = n_base_filters
        self.output_nc = output_nc
        self.gpu_ids = gpu_ids
        self.norm_layer = norm_layer
        self.n_blocks = n_blocks

        self.reflection1 = nn.ReflectionPad2d(1)    # for resnet
        self.reflection3 = nn.ReflectionPad2d(3)
        self.constantpad1 = nn.ConstantPad2d(1, 0)  # for resnet
        self.constantpad3 = nn.ConstantPad2d(3, 0)

        self.norm_layer_mask = self.norm_layer(n_sparse_filters)    # 16
        self.norm_layer1 = self.norm_layer(n_base_filters)          # 64
        self.norm_layer2 = self.norm_layer(2 * n_base_filters)      # 128
        self.norm_layer3 = self.norm_layer(4 * n_base_filters)      # 256
        self.norm_layer4 = self.norm_layer(8 * n_base_filters)      # 512

        self.conv1 = nn.Conv2d(in_channels=input_nc+input_sparse_nc,
                               out_channels=n_base_filters,
                               kernel_size=7, stride=1, padding=0)

        self.conv2 = nn.Conv2d(in_channels=n_base_filters + n_sparse_filters,
                               out_channels=2 * n_base_filters,
                               kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(in_channels=2 * n_base_filters + n_sparse_filters,
                               out_channels=4 * n_base_filters,
                               kernel_size=3, stride=2, padding=1)

        self.conv4 = nn.Conv2d(in_channels=4 * n_base_filters + n_sparse_filters,
                               out_channels=8 * n_base_filters,
                               kernel_size=3, stride=2, padding=1)

        self.sparse_conv1 = SparseConv(in_channels=input_sparse_nc,
                                       out_channels=n_sparse_filters,
                                       kernel_size=7, stride=1, padding=0)

        self.sparse_conv2 = SparseConv(in_channels=n_sparse_filters,
                                       out_channels=n_sparse_filters,
                                       kernel_size=3, stride=2, padding=1)

        self.sparse_conv3 = SparseConv(in_channels=n_sparse_filters,
                                       out_channels=n_sparse_filters,
                                       kernel_size=3, stride=2, padding=1)

        self.sparse_conv4 = SparseConv(in_channels=n_sparse_filters,
                                       out_channels=n_sparse_filters,
                                       kernel_size=3, stride=2, padding=1)

        self.resNet_conv1 = nn.Conv2d(in_channels=8 * n_base_filters + n_sparse_filters,
                                      out_channels=8 * n_base_filters,
                                      kernel_size=3, stride=1, padding=0)

        self.resNet_conv2 = nn.Conv2d(in_channels=8 * n_base_filters + n_sparse_filters,
                                      out_channels=8 * n_base_filters,
                                      kernel_size=3, stride=1, padding=0)

        self.resNet_sparse_conv1 = SparseConv(in_channels=n_sparse_filters,
                                              out_channels=16, kernel_size=3,
                                              stride=1, padding=0)

        self.resNet_sparse_conv2 = SparseConv(in_channels=n_sparse_filters,
                                              out_channels=16, kernel_size=3,
                                              stride=1, padding=0)

        self.relu = nn.ReLU(True)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.convTran = nn.Sequential(
            nn.ConvTranspose2d(n_base_filters * 8, n_base_filters * 4,
                               kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=use_bias),
            norm_layer(n_base_filters * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(n_base_filters * 4, n_base_filters * 2,
                               kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=use_bias),
            norm_layer(n_base_filters * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(n_base_filters * 2, n_base_filters,
                               kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=use_bias),
            norm_layer(n_base_filters),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(n_base_filters, output_nc,
                      kernel_size=7, stride=1, padding=0),
            # nn.ReLU(inplace=True)
            # nn.Sigmoid()
            nn.Tanh()
        )

    def conv(self, input_a, input_c, mask):

        # layer1
        input_a = self.reflection3(input_a)  # padding
        input_c = self.constantpad3(input_c)
        mask = self.constantpad3(mask)

        feature_a = _full_conv(self.conv1, input_a, input_c, mask)
        feature_c, feature_mask = self.sparse_conv1(input_c, mask)
        feature_a = self.norm_layer1(feature_a)
        feature_c = self.norm_layer_mask(feature_c)
        feature_a = self.relu(feature_a)
        feature_c = self.relu(feature_c)

        # layer2
        feature_a = _full_conv(self.conv2, feature_a, feature_c, feature_mask)
        feature_c, feature_mask = self.sparse_conv2(feature_c, feature_mask)
        feature_a = self.norm_layer2(feature_a)
        feature_c = self.norm_layer_mask(feature_c)
        feature_a = self.relu(feature_a)
        feature_c = self.relu(feature_c)

        # layer3
        feature_a = _full_conv(self.conv3, feature_a, feature_c, feature_mask)
        feature_c, feature_mask = self.sparse_conv3(feature_c, feature_mask)
        feature_a = self.norm_layer3(feature_a)
        feature_c = self.norm_layer_mask(feature_c)
        feature_a = self.relu(feature_a)
        feature_c = self.relu(feature_c)

        # layer4
        feature_a = _full_conv(self.conv4, feature_a, feature_c, feature_mask)
        feature_c, feature_mask = self.sparse_conv4(feature_c, feature_mask)
        feature_a = self.norm_layer4(feature_a)
        feature_c = self.norm_layer_mask(feature_c)
        feature_a = self.relu(feature_a)
        feature_c = self.relu(feature_c)

        # layer4--resNet
        for i in range(self.n_blocks):

            feature_tmp = feature_a  # store original feature before resNet
            feature_mask_tmp = feature_mask

            # resnet_layer1
            feature_a = self.reflection1(feature_a)
            feature_c = self.constantpad1(feature_c)
            feature_mask = self.constantpad1(feature_mask)

            feature_a = _full_conv(self.resNet_conv1, feature_a, feature_c, feature_mask)
            feature_c, feature_mask = self.resNet_sparse_conv1(feature_c, feature_mask)
            feature_a = self.norm_layer4(feature_a)
            feature_c = self.norm_layer_mask(feature_c)
            feature_a = self.relu(feature_a)
            feature_c = self.relu(feature_c)

            # resnet_layer2
            feature_a = self.reflection1(feature_a)
            feature_c = self.constantpad1(feature_c)
            feature_mask = self.constantpad1(feature_mask)

            feature_a = _full_conv(self.resNet_conv2, feature_a, feature_c, feature_mask)
            feature_c, feature_mask = self.resNet_sparse_conv2(feature_c, feature_mask)
            feature_a = self.norm_layer4(feature_a)
            feature_c = self.norm_layer_mask(feature_c)

            feature_a = feature_tmp + feature_a  # x = x +resNet(x)
            feature_mask = feature_mask_tmp + feature_mask

        feature_a = self.convTran(feature_a)

        return feature_a

    def forward(self, input_a, input_c, mask):

        feature_a = self.conv(input_a, input_c, mask)

        return feature_a
