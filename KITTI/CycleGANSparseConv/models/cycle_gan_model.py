# -*- coding:utf-8 -*-
import os
import torch
import itertools
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
        self.netG_B = None

        self.netD_A = None
        self.netD_B = None

        self.criterionGAN = None
        self.optimizer_G = None

        self.optimizer_D_A = None
        self.optimizer_D_B = None

        self.optimizers = []
        self.schedulers = []

        self.image_paths = None

        self.real_a = None
        self.real_b = None
        self.real_c = None
        self.rec_a = None
        self.rec_b = None
        self.fake_a = None
        self.fake_b = None

        self.fake_a_pool = None
        self.fake_b_pool = None

        self.loss_cycle_A = None
        self.loss_cycle_B = None
        self.loss_D_A = None
        self.loss_D_B = None
        self.loss_G = None
        self.loss_g_a = None
        self.loss_g_b = None
        self.loss_idt_A = None
        self.loss_idt_B = None
        self.loss_sparse_C = None
        self.fake_a_all_loss = None
        self.fake_b_all_loss = None

        self.mask_one = None
        self.a_mask = None
        self.b_mask = None
        self.c_mask = None

        self.sparse_B = None

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
        self.netG_B = networks.define_g_b(1, 3, opt.norm, opt.init_type, self.gpu_ids)

        if self.isTrain:
            self.netD_A = networks.define_d(1, 64, opt.norm, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_d(3, 64, opt.norm, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:

            # define loss functions
            self.criterionGAN = networks.GANLoss(tensor=self.Tensor)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(0.5, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer))

        # print('---------- Networks initialized -------------')
        # networks.print_network(self.netG_A)
        # networks.print_network(self.netG_B)
        # if self.isTrain:
        #     networks.print_network(self.netD_A)
        #     networks.print_network(self.netD_B)
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
        self.rec_a = self.netG_B(fake_b).data
        self.fake_b = fake_b.data

        fake_a = self.netG_B(real_b)
        self.rec_b = self.netG_A(fake_a, real_c, self.c_mask).data
        self.fake_a = fake_a.data

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

    def backward_d_basic(self, net_d, real, fake):

        pred_real = net_d(real)
        loss_d_real = self.criterionGAN(pred_real, True)

        pred_fake = net_d(fake.detach())
        loss_d_fake = self.criterionGAN(pred_fake, False)

        loss_d = (loss_d_real + loss_d_fake) * 0.5

        loss_d.backward()

        return loss_d

    def backward_d_a(self):

        fake_b = Variable(self.fake_b)
        loss_d_a = self.backward_d_basic(self.netD_A, self.real_b, fake_b)
        self.loss_D_A = loss_d_a.data[0]

    def backward_d_b(self):

        fake_a = Variable(self.fake_a)
        loss_d_b = self.backward_d_basic(self.netD_B, self.real_a, fake_a)
        self.loss_D_B = loss_d_b.data[0]

    def backward_g(self):

        lambda_a = self.opt.lambda_A
        lambda_b = self.opt.lambda_B

        self.a_mask = self.get_mask(self.real_a)
        self.b_mask = self.get_mask(self.real_b)
        self.c_mask = self.get_mask(self.real_c)

        fake_b = self.netG_A(self.real_a, self.real_c, self.c_mask)
        # torch.save(fake_b, "./fake_b.p")

        pred_fake = self.netD_A(fake_b)
        loss_g_a = self.criterionGAN(pred_fake, True)

        fake_a = self.netG_B(self.real_b)
        pred_fake = self.netD_B(fake_a)
        loss_g_b = self.criterionGAN(pred_fake, True)

        rec_a = self.netG_B(fake_b)
        loss_cycle_a = torch.nn.L1Loss()(rec_a, self.real_a)

        rec_b = self.netG_A(fake_a, self.real_c, self.c_mask)
        loss_cycle_b = self.calculate_l1_loss(rec_b, self.real_b, self.b_mask)
        
        # change to MSE_LOSS
        loss_sparse_c = self.calculate_rmse_loss(fake_b, self.real_c, self.c_mask)

        fake_a_all_loss = torch.nn.L1Loss()(fake_a, self.real_a)
        fake_b_all_loss = self.calculate_l1_loss(fake_b, self.real_b, self.b_mask)

        loss_g = (loss_g_a + loss_g_b + loss_cycle_a * lambda_a + loss_cycle_b * lambda_b
                  + loss_sparse_c * self.opt.sparse_k)

        loss_g.backward()

        self.fake_b = fake_b.data
        self.fake_a = fake_a.data
        self.rec_a = rec_a.data
        self.rec_b = rec_b.data

        self.loss_g_a = loss_g_a.data[0]
        self.loss_g_b = loss_g_b.data[0]
        self.loss_cycle_A = loss_cycle_a.data[0]
        self.loss_cycle_B = loss_cycle_b.data[0]
        self.loss_sparse_C = loss_sparse_c.data[0]
        self.fake_a_all_loss = fake_a_all_loss.data[0]
        self.fake_b_all_loss = fake_b_all_loss.data[0]
        self.loss_G = loss_g.data[0]

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_g()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_d_a()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_d_b()
        self.optimizer_D_B.step()

    def get_current_errors(self):

        ret_errors = OrderedDict([('fake_A_all_loss', self.fake_a_all_loss), ('fake_B_all_loss', self.fake_b_all_loss),
                                  ('Sparse_C', self.loss_sparse_C),
                                  ('D_A', self.loss_D_A), ('G_A', self.loss_g_a), ('Cyc_A', self.loss_cycle_A),
                                  ('D_B', self.loss_D_B), ('G_B', self.loss_g_b), ('Cyc_B',  self.loss_cycle_B),
                                  ('loss_G', self.loss_G)])

        return ret_errors, self.loss_sparse_C

    def get_current_visuals(self):
        real_a = util.tensor2im(self.input_a)
        fake_b = util.tensor2im(self.fake_b)
        rec_a = util.tensor2im(self.rec_a)
        real_b = util.tensor2im(self.input_b)
        fake_a = util.tensor2im(self.fake_a)
        rec_b = util.tensor2im(self.rec_b)
        real_c = util.tensor2im(self.input_c)

        ret_visuals = OrderedDict([('real_a', real_a), ('fake_b', fake_b), ('rec_a', rec_a), ('real_c', real_c),
                                   ('real_b', real_b), ('fake_a', fake_a), ('rec_b', rec_b)])

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
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

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
