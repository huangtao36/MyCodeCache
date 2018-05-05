# -*- coding:utf-8 -*-
from collections import OrderedDict
import itertools
import utilSet.util as util
from . import networks
import torch
from torch.autograd import Variable
import cv2
import os


class CycleGANModel:
    
    def __init__(self):
        super(CycleGANModel, self).__init__()

        self.input_A = None
        self.input_B = None

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
        self.criterionCycle = None
        self.criterionIdt = None
        self.optimizer_G = None
        self.optimizer_D_A = None
        self.optimizer_D_B = None

        self.optimizers = []
        self.schedulers = []

        self.image_paths = None

        self.real_a = None
        self.real_b = None
        self.rec_a = None
        self.rec_b = None
        self.fake_a = None
        self.fake_b = None
        self.idt_A = None
        self.idt_B = None

        self.loss_cycle_A = None
        self.loss_cycle_B = None
        self.loss_D_A = None
        self.loss_D_B = None
        self.loss_G = None
        self.loss_G_A = None
        self.loss_G_B = None
        self.loss_idt_A = None
        self.loss_idt_B = None

        self.fake_a_all_loss = None
        self.fake_b_all_loss = None
        self.fake_b_lr_loss = None

        self.old_lr = None

    def initialize(self, opt):

        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor
        self.save_dir = os.path.join(opt.result_root_dir, opt.variable, opt.variable_value, 'Net')
        util.mkdirs(self.save_dir)

        self.netG_A = networks.define_g(3 + 1, 3, 64, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_g(3, 3, 64, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_d(3, 64, opt.norm, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_d(3, 64, opt.norm, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.criterionGAN = networks.GANLoss(tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(0.5, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        # print('---------- Networks initialized -------------')
        # networks.print_network(self.netG_A)
        # networks.print_network(self.netG_B)
        # if self.isTrain:
        #     networks.print_network(self.netD_A)
        #     networks.print_network(self.netD_B)
        # print('-----------------------------------------------')

    def set_input(self, input_):
        self.input_A = input_['A']
        self.input_B = input_['B']

        self.image_paths = input_['A_paths']  # 用于给输出图片获得原始文件名标签，

    def forward(self):
        self.real_a = Variable(self.input_A).cuda()
        self.real_b = Variable(self.input_B).cuda()

    def test(self):
        real_a = Variable(self.input_A, volatile=True).cuda()
        real_b = Variable(self.input_B, volatile=True).cuda()

        grid_b = self.get_fuse_layer(real_b)

        fake_b = self.netG_A(self.fuse(real_a, grid_b))
        self.rec_a = self.netG_B(fake_b).data
        self.fake_b = fake_b.data

        fake_a = self.netG_B(real_b)
        self.rec_b = self.netG_A(self.fuse(fake_a, grid_b)).data
        self.fake_a = fake_a.data

    def get_image_paths(self):
        return self.image_paths

    def get_fuse_layer(self, real):

        grid = real[:, :, ::self.opt.step_size, ::self.opt.step_size].data

        grid_npy = grid[0, :, :, :].cpu().numpy()
        grid_trans = grid_npy.transpose(1, 2, 0)
        resize = cv2.resize(grid_trans, (256, 256), interpolation=cv2.INTER_CUBIC)
        resize[::self.opt.step_size, ::self.opt.step_size, :] = grid_trans
        resize_trans = resize.transpose(2, 0, 1)
        full = Variable(torch.from_numpy(resize_trans[0, :, :])).cuda()
        full = torch.unsqueeze(torch.unsqueeze(full, 0), 0)
        return full

    @staticmethod
    def fuse(in_put, full):
        img_cat = torch.cat([in_put.data, full.data], dim=1)
        return Variable(img_cat).type(torch.cuda.FloatTensor).cuda()

    def backward_d_basic(self, net_d, real, fake):
        predicted_real = net_d(real)
        loss_d_real = self.criterionGAN(predicted_real, True)
        predicted_fake = net_d(fake.detach())
        loss_d_fake = self.criterionGAN(predicted_fake, False)
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
        lambda_idt = self.opt.identity
        lambda_a = self.opt.lambda_A
        lambda_b = self.opt.lambda_B

        grid_b = self.get_fuse_layer(self.real_b)

        if lambda_idt > 0:
            idt_a = self.netG_A(self.fuse(self.real_b, grid_b))
            loss_idt_a = self.criterionIdt(idt_a, self.real_b) * lambda_b * lambda_idt
            idt_b = self.netG_B(self.real_a)
            loss_idt_b = self.criterionIdt(idt_b, self.real_a) * lambda_a * lambda_idt

            self.idt_A = idt_a.data
            self.idt_B = idt_b.data
            self.loss_idt_A = loss_idt_a.data[0]
            self.loss_idt_B = loss_idt_b.data[0]
        else:
            loss_idt_a = 0
            loss_idt_b = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        fake_b = self.netG_A(self.fuse(self.real_a, grid_b))
        predicted_fake = self.netD_A(fake_b)
        loss_g_a = self.criterionGAN(predicted_fake, True)

        fake_a = self.netG_B(self.real_b)
        predicted_fake = self.netD_B(fake_a)
        loss_g_b = self.criterionGAN(predicted_fake, True)

        rec_a = self.netG_B(fake_b)
        loss_cycle_a = self.criterionCycle(rec_a, self.real_a) * lambda_a

        rec_b = self.netG_A(self.fuse(fake_a, grid_b))
        loss_cycle_b = self.criterionCycle(rec_b, self.real_b) * lambda_b

        fake_b_grid = fake_b[:, :, ::self.opt.step_size, ::self.opt.step_size]
        real_b_grid = self.real_b[:, :, ::self.opt.step_size, ::self.opt.step_size]
        fake_b_lr_loss = torch.nn.L1Loss()(fake_b_grid, real_b_grid)

        fake_a_all_loss = torch.nn.L1Loss()(fake_a, self.real_a)
        fake_b_all_loss = torch.nn.L1Loss()(fake_b, self.real_b)

        loss_g = (loss_g_a + loss_g_b + loss_cycle_a + loss_cycle_b +
                  loss_idt_a + loss_idt_b + fake_b_lr_loss * self.opt.sparse_k)
        loss_g.backward()

        self.fake_b = fake_b.data
        self.fake_a = fake_a.data
        self.rec_a = rec_a.data
        self.rec_b = rec_b.data

        self.loss_G_A = loss_g_a.data[0]
        self.loss_G_B = loss_g_b.data[0]
        self.loss_cycle_A = loss_cycle_a.data[0]
        self.loss_cycle_B = loss_cycle_b.data[0]
        self.fake_b_lr_loss = fake_b_lr_loss.data[0]
        self.fake_a_all_loss = fake_a_all_loss.data[0]
        self.fake_b_all_loss = fake_b_all_loss.data[0]
        self.loss_G = loss_g.data[0]

    def optimize_parameters(self):

        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_g()
        self.optimizer_G.step()

        self.optimizer_D_A.zero_grad()
        self.backward_d_a()
        self.optimizer_D_A.step()

        self.optimizer_D_B.zero_grad()
        self.backward_d_b()
        self.optimizer_D_B.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('fake_A_all_loss', self.fake_a_all_loss), ('fake_B_all_loss', self.fake_b_all_loss),
                                  ('fake_B_lr_loss', self.fake_b_lr_loss),
                                  ('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Cyc_A', self.loss_cycle_A),
                                  ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Cyc_B',  self.loss_cycle_B),
                                  ('loss_G', self.loss_G)])
        if self.opt.identity > 0.0:
            ret_errors['idt_A'] = self.loss_idt_A
            ret_errors['idt_B'] = self.loss_idt_B
        return ret_errors

    def get_current_visuals(self):
        real_a = util.tensor2im(self.input_A)
        fake_b = util.tensor2im(self.fake_b)
        rec_a = util.tensor2im(self.rec_a)
        real_b = util.tensor2im(self.input_B)
        fake_a = util.tensor2im(self.fake_a)
        rec_b = util.tensor2im(self.rec_b)
        
        ret_visuals = OrderedDict([('real_A', real_a), ('fake_B', fake_b), ('rec_A', rec_a),
                                   ('real_B', real_b), ('fake_A', fake_a), ('rec_B', rec_b)])
        
        if self.opt.isTrain and self.opt.identity > 0.0:
            ret_visuals['idt_A'] = util.tensor2im(self.idt_A)
            ret_visuals['idt_B'] = util.tensor2im(self.idt_B)
            
        return ret_visuals

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

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
