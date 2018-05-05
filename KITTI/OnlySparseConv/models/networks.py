# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from .Sparse_conv import SparseConv
from config import TrainOptions

opt = TrainOptions().parse()


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


def define_g(input_nc, output_nc, norm='instance', init_type='xavier', gpu_ids=[]):

    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    net_g = Generate(input_nc=input_nc, output_nc=output_nc, norm_layer=norm_layer, gpu_ids=gpu_ids)

    if len(gpu_ids) > 0:
        net_g.cuda(gpu_ids[0])

    init_weights(net_g, init_type=init_type)

    return net_g


class Generate(nn.Module):
    def __init__(self, input_nc=1, output_nc=1,
                 n_base_filters=64,
                 norm_layer=nn.BatchNorm2d,
                 n_blocks=9, gpu_ids=[]):

        assert (n_blocks >= 0)
        super(Generate, self).__init__()

        self.n_base_filters = n_base_filters
        self.output_nc = output_nc
        self.gpu_ids = gpu_ids
        self.norm_layer = norm_layer
        self.n_blocks = n_blocks

        self.constantpad1 = nn.ConstantPad2d(1, 0)
        self.constantpad3 = nn.ConstantPad2d(3, 0)

        self.norm_layer1 = self.norm_layer(n_base_filters)          # 64
        self.norm_layer2 = self.norm_layer(2 * n_base_filters)      # 128
        self.norm_layer3 = self.norm_layer(4 * n_base_filters)      # 256
        self.norm_layer4 = self.norm_layer(8 * n_base_filters)      # 512

        self.sparse_conv1 = SparseConv(in_channels=input_nc,
                                       out_channels=n_base_filters,
                                       kernel_size=7, stride=1, padding=0)

        self.sparse_conv2 = SparseConv(in_channels=n_base_filters,
                                       out_channels=2 * n_base_filters,
                                       kernel_size=3, stride=2, padding=1)

        self.sparse_conv3 = SparseConv(in_channels=2 * n_base_filters,
                                       out_channels=4 * n_base_filters,
                                       kernel_size=3, stride=2, padding=1)

        self.sparse_conv4 = SparseConv(in_channels=4 * n_base_filters,
                                       out_channels=8 * n_base_filters,
                                       kernel_size=3, stride=2, padding=1)

        self.resNet_sparse_conv1 = SparseConv(in_channels=8 * n_base_filters,
                                              out_channels=8 * n_base_filters, kernel_size=3,
                                              stride=1, padding=0)

        self.resNet_sparse_conv2 = SparseConv(in_channels=8 * n_base_filters,
                                              out_channels=8 * n_base_filters, kernel_size=3,
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

            nn.Tanh()
        )

    def forward(self, input_c, mask):

        # layer1
        input_c = self.constantpad3(input_c)
        mask = self.constantpad3(mask)

        feature_c, feature_mask = self.sparse_conv1(input_c, mask)
        feature_c = self.norm_layer1(feature_c)
        feature_c = self.relu(feature_c)

        # layer2
        feature_c, feature_mask = self.sparse_conv2(feature_c, feature_mask)
        feature_c = self.norm_layer2(feature_c)
        feature_c = self.relu(feature_c)

        # layer3
        feature_c, feature_mask = self.sparse_conv3(feature_c, feature_mask)
        feature_c = self.norm_layer3(feature_c)
        feature_c = self.relu(feature_c)

        # layer4
        feature_c, feature_mask = self.sparse_conv4(feature_c, feature_mask)
        feature_c = self.norm_layer4(feature_c)
        feature_c = self.relu(feature_c)

        # layer4--resNet
        for i in range(self.n_blocks):
            feature_c_tmp = feature_c  # store original feature before resNet

            # resnet_layer1
            feature_c = self.constantpad1(feature_c)
            feature_mask = self.constantpad1(feature_mask)

            feature_c, feature_mask = self.resNet_sparse_conv1(feature_c, feature_mask)
            feature_c = self.norm_layer4(feature_c)
            feature_c = self.relu(feature_c)

            # resnet_layer2
            feature_c = self.constantpad1(feature_c)
            feature_mask = self.constantpad1(feature_mask)

            feature_c, feature_mask = self.resNet_sparse_conv2(feature_c, feature_mask)
            feature_c = self.norm_layer4(feature_c)

            feature_c = feature_c_tmp + feature_c

        # up_sampling
        feature_c = self.convTran(feature_c)

        return feature_c


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
