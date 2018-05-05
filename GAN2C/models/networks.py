# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler


def weights_init_xavier(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif class_name.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif class_name.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    if init_type == 'xavier':
        net.apply(weights_init_xavier)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


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


def get_scheduler(optimizer, opt):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler


def define_g(input_nc, output_nc, ngf, norm='instance', use_dropout=False, init_type='xavier', gpu_ids=[]):

    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    net_g = ResnetGenerator(input_nc, output_nc, ngf,
                            norm_layer=norm_layer, use_dropout=use_dropout,
                            n_blocks=1, gpu_ids=gpu_ids)

    if len(gpu_ids) > 0:
        net_g.cuda(gpu_ids[0])
    init_weights(net_g, init_type=init_type)
    return net_g


def define_d(input_nc, ndf, norm='instance', init_type='xavier', gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    net_d = NLayerDiscriminator(input_nc, ndf, n_layers=3,
                                norm_layer=norm_layer,
                                gpu_ids=gpu_ids)

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

    def get_target_tensor(self, in_put, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != in_put.numel()))
            if create_label:
                real_tensor = self.Tensor(in_put.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != in_put.numel()))
            if create_label:
                fake_tensor = self.Tensor(in_put.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, in_put, target_is_real):
        target_tensor = self.get_target_tensor(in_put, target_is_real)
        return self.loss(in_put, target_tensor)      # MSELoss


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=1, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2,
                                kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                  norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, in_put):
        if self.gpu_ids and isinstance(in_put.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, in_put, self.gpu_ids)
        else:
            return self.model(in_put)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()

        self.conv_block = self.build_conv_block(dim, norm_layer, use_bias)

    def build_conv_block(self, dim, norm_layer, use_bias):
        conv_block = []
        p = 0
        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]

        conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


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
