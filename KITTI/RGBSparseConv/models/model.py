# -*- coding:utf-8 -*-
import os
import torch
from . import networks
import numpy as np
import utilSet.util as util
from torch.autograd import Variable
from collections import OrderedDict
from evaluate.depth_evaluate import calculate_depth_error


class CycleGANModel:

    def __init__(self):
        super(CycleGANModel, self).__init__()

        self.input_a = None  # RGB
        self.input_b = None  # groundtruth
        self.input_c = None  # sparse

        self.opt = None
        self.gpu_ids = None
        self.isTrain = None
        self.Tensor = None
        self.save_dir = None

        self.netG_A = None

        self.optimizer_G = None

        self.optimizers = []
        self.schedulers = []

        self.image_paths = None

        self.real_a = None
        self.real_b = None
        self.real_c = None
        self.fake_b = None

        self.loss_G = None
        self.sparse_loss = None
        self.fake_b_loss = None

        self.mask_one = None
        self.a_mask = None
        self.b_mask = None
        self.c_mask = None

    def initialize(self, opt):

        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor
        self.save_dir = os.path.join(opt.result_root_dir, opt.variable, opt.variable_value, 'Net')
        util.mkdirs(self.save_dir)

        mask_one = torch.ones((opt.height, opt.width))
        self.mask_one = torch.unsqueeze(torch.unsqueeze(mask_one, 0), 0)
        self.mask_one = Variable(self.mask_one).cuda()

        self.netG_A = networks.define_g_a(3, 1, 1, opt.norm, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)

        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer))

        # print('---------- Networks initialized -------------')
        # networks.print_network(self.netG_A)
        # print('-----------------------------------------------')

    def set_input(self, input_):

        self.input_a = input_['A']
        self.input_b = input_['B']
        self.input_c = input_['C']

        self.image_paths = input_['A_paths']  # 用于给输出图片获得原始文件名

    def forward(self):

        self.real_a = Variable(self.input_a).cuda()
        self.real_b = Variable(self.input_b).cuda()
        self.real_c = Variable(self.input_c).cuda()

    def test(self):

        real_a = Variable(self.input_a, volatile=True).cuda()
        real_b = Variable(self.input_b, volatile=True).cuda()
        real_c = Variable(self.input_c, volatile=True).cuda()

        self.a_mask = self.get_mask(real_a)
        self.b_mask = self.get_mask(real_b)
        self.c_mask = self.get_mask(real_c)

        fake_b = self.netG_A(real_a, real_c, self.c_mask)
        self.fake_b = fake_b.data

    def get_image_paths(self):
        return self.image_paths

    @staticmethod
    def get_mask(matrix):

        if matrix.size()[1] == 3:
            matrix = matrix[:, 0, :, :] + matrix[:, 1, :, :] + matrix[:, 2, :, :]
            matrix = torch.unsqueeze(matrix, 1)

        mask = matrix > 0
        return mask.float()

    @staticmethod
    def calculate_l1_loss(a, b, mask):

        count = np.count_nonzero(mask.cpu().data.numpy())
        sum_loss = torch.sum(torch.abs(a * mask - b * mask))

        loss = sum_loss/count

        return loss

    @staticmethod
    def calculate_rmse_loss(a, b, mask):

        count = np.count_nonzero(mask.cpu().data.numpy())
        
        sum_loss = torch.sum((a * mask - b * mask)**2)

        loss = sum_loss / count
        loss = torch.sqrt(loss)

        return loss

    def backward_g(self):

        self.a_mask = self.get_mask(self.real_a)
        self.b_mask = self.get_mask(self.real_b)
        self.c_mask = self.get_mask(self.real_c)

        fake_b = self.netG_A(self.real_a, self.real_c, self.c_mask)
        
        # change to MSE_LOSS
        sparse_loss = self.calculate_rmse_loss(fake_b, self.real_c, self.c_mask)

        fake_b_loss = self.calculate_l1_loss(fake_b, self.real_b, self.b_mask)

        loss_g = (sparse_loss * self.opt.sparse_k + fake_b_loss * self.opt.fake_b_loss_k)

        loss_g.backward()

        self.fake_b = fake_b.data

        self.sparse_loss = sparse_loss.data[0]
        self.fake_b_loss = fake_b_loss.data[0]
        self.loss_G = loss_g.data[0]

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_g()
        self.optimizer_G.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('Sparse_C', self.sparse_loss),
                                  ('fake_b_loss', self.fake_b_loss),
                                  ('loss_G', self.loss_G)])

        return ret_errors

    def get_current_visuals(self):
        real_a = util.tensor2im(self.input_a)
        fake_b = util.tensor2im(self.fake_b)
        real_b = util.tensor2im(self.input_b)
        real_c = util.tensor2im(self.input_c)

        ret_visuals = OrderedDict([('real_a', real_a), ('fake_b', fake_b), ('real_b', real_b), ('real_c', real_c)])

        return ret_visuals

    def get_depth_errors(self):
        groundtruth = self.input_b[0, 0, :, :].cpu().float().numpy()
        groundtruth = groundtruth * 85

        predict = self.fake_b[0, 0, :, :].cpu().float().numpy()
        predict = predict * 85

        irmse, imae, rmse, mae = calculate_depth_error(groundtruth, predict)
        depth_errors = OrderedDict([('iRMSE', irmse), ('iMAE', imae), ('RMSE_m', rmse), ('MAE_m', mae)])

        return depth_errors

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
