# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
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


# 生成器
def define_g_b(input_nc, output_nc, norm='instance', init_type='xavier', gpu_ids=[]):

    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    net_g_b = GenerateB(input_nc=input_nc, output_nc=output_nc, norm_layer=norm_layer, gpu_ids=gpu_ids)

    if len(gpu_ids) > 0:
        net_g_b.cuda(gpu_ids[0])

    init_weights(net_g_b, init_type=init_type)

    return net_g_b


# 判别器定义
def define_d(input_nc, ndf, norm='instance', init_type='xavier', gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    net_d = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, gpu_ids=gpu_ids)

    if use_gpu:
        net_d.cuda(gpu_ids[0])

    init_weights(net_d, init_type=init_type)

    return net_d


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class GANLoss(nn.Module):

    def __init__(self, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()

        self.real_label = 1.0
        self.fake_label = 0.0
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor

        self.loss = nn.MSELoss()

    def get_target_tensor(self, input_matrix, target_is_real):

        if target_is_real:
            # 创建一个与输入相同size，每个值均为1的tensor矩阵变量
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input_matrix.numel()))
            if create_label:
                real_tensor = self.Tensor(input_matrix.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            # 创建一个与输入相同size，每个值均为0的tensor矩阵变量
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input_matrix.numel()))
            if create_label:
                fake_tensor = self.Tensor(input_matrix.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input_matrix, target_is_real):
        target_tensor = self.get_target_tensor(input_matrix, target_is_real)
        return self.loss(input_matrix, target_tensor)      # MSELoss

    def forward(self):
        pass


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


class GenerateB(nn.Module):
    def __init__(self, input_nc=3, n_base_filters=64, output_nc=3,
                 norm_layer=nn.BatchNorm2d, n_blocks=9, gpu_ids=[]):

        assert (n_blocks >= 0)
        super(GenerateB, self).__init__()

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

        self.norm_layer1 = self.norm_layer(n_base_filters)          # 64
        self.norm_layer2 = self.norm_layer(2 * n_base_filters)      # 128
        self.norm_layer3 = self.norm_layer(4 * n_base_filters)      # 256
        self.norm_layer4 = self.norm_layer(8 * n_base_filters)      # 512

        self.conv1 = nn.Conv2d(in_channels=input_nc,
                               out_channels=n_base_filters,
                               kernel_size=7, stride=1, padding=0)

        self.conv2 = nn.Conv2d(in_channels=n_base_filters,
                               out_channels=2 * n_base_filters,
                               kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(in_channels=2 * n_base_filters,
                               out_channels=4 * n_base_filters,
                               kernel_size=3, stride=2, padding=1)

        self.conv4 = nn.Conv2d(in_channels=4 * n_base_filters,
                               out_channels=8 * n_base_filters,
                               kernel_size=3, stride=2, padding=1)

        self.resNet_conv1 = nn.Conv2d(in_channels=8 * n_base_filters,
                                      out_channels=8 * n_base_filters,
                                      kernel_size=3, stride=1, padding=0)

        self.resNet_conv2 = nn.Conv2d(in_channels=8 * n_base_filters,
                                      out_channels=8 * n_base_filters,
                                      kernel_size=3, stride=1, padding=0)

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

    def conv(self, input_a):

        # layer1
        input_a = self.reflection3(input_a)  # padding

        feature_a = self.conv1(input_a)
        feature_a = self.norm_layer1(feature_a)
        feature_a = self.relu(feature_a)

        # layer2
        feature_a = self.conv2(feature_a)
        feature_a = self.norm_layer2(feature_a)
        feature_a = self.relu(feature_a)

        # layer3
        feature_a = self.conv3(feature_a)
        feature_a = self.norm_layer3(feature_a)
        feature_a = self.relu(feature_a)

        # layer4
        feature_a = self.conv4(feature_a)
        feature_a = self.norm_layer4(feature_a)
        feature_a = self.relu(feature_a)

        # layer4--resNet
        for i in range(self.n_blocks):

            feature_tmp = feature_a  # store original feature before resNet

            # resnet_layer1
            feature_a = self.reflection1(feature_a)

            feature_a = self.resNet_conv1(feature_a)
            feature_a = self.norm_layer4(feature_a)
            feature_a = self.relu(feature_a)

            # resnet_layer2
            feature_a = self.reflection1(feature_a)

            feature_a = self.resNet_conv2(feature_a)
            feature_a = self.norm_layer4(feature_a)

            feature_a = feature_tmp + feature_a  # x = x +resNet(x)
            # feature_a = self.relu(feature_a)

        feature_a = self.convTran(feature_a)

        return feature_a

    def forward(self, input_a):

        feature_a = self.conv(input_a)

        return feature_a


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, True)]  # do no use InstanceNorm

        nf_mult = 1
        for n in range(1, n_layers+1):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            if n < n_layers:   # n=1,2-->stride=2   n=3-->stride=1
                sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                       kernel_size=4, stride=2, padding=1, bias=use_bias),
                             norm_layer(ndf * nf_mult),
                             nn.LeakyReLU(0.2, True)]
            else:
                sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                       kernel_size=4, stride=1, padding=1, bias=use_bias),
                             norm_layer(ndf * nf_mult),
                             nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input_):
        if len(self.gpu_ids) and isinstance(input_.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input_, self.gpu_ids)
        else:
            return self.model(input_)
