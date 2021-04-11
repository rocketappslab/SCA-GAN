import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
# losses
from losses.L1_plus_perceptualLoss import L1_plus_perceptualLoss
from losses.CX_style_loss import CXLoss
from .vgg import VGG


class TransferModel(BaseModel):
    def name(self):
        return 'TransferModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        input_nc = 3
        self.netG = networks.define_G(opt, input_nc, 1,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                        n_downsampling=opt.G_n_downsampling)


        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if opt.with_D_PB:
                input_nc = 18 + 3
                self.netD_PB = networks.define_D(input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

            if opt.with_D_PP:
                input_nc = 3 + 3
                self.netD_PP = networks.define_D(input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

            # if len(opt.gpu_ids) > 1:
            #     self.load_EdgePriorNetwork(self.netG.module)
            # else:
            #     self.load_EdgePriorNetwork(self.netG)
            if len(opt.gpu_ids) > 1:
                self.load_EdgePriorNetwork(self.netG.module.edge_generator.model_edge)
            else:
                self.load_EdgePriorNetwork(self.netG.edge_generator.model_edge)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'netG', which_epoch)
            if self.isTrain:
                if opt.with_D_PB:
                    self.load_network(self.netD_PB, 'netD_PB', which_epoch)
                if opt.with_D_PP:
                    self.load_network(self.netD_PP, 'netD_PP', which_epoch)


        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_PP_pool = ImagePool(opt.pool_size)
            self.fake_PB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)

            if opt.L1_type == 'origin':
                self.criterionL1 = torch.nn.L1Loss()
            elif opt.L1_type == 'l1_plus_perL1':
                self.criterionL1 = L1_plus_perceptualLoss(opt.lambda_A, opt.lambda_B, opt.perceptual_layers, self.gpu_ids, opt.percep_is_l1)
            else:
                raise Exception('Unsurportted type of L1!')

            if opt.use_cxloss:
                self.CX_loss = CXLoss(sigma=0.5)
                if torch.cuda.is_available():
                    self.CX_loss.cuda()
                self.vgg = VGG()
                self.vgg.load_state_dict(torch.load('fashion_data/vgg_conv.pth'))
                for param in self.vgg.parameters():
                    param.requires_grad = False
                if torch.cuda.is_available():
                    self.vgg.cuda()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))


            if opt.with_D_PB:
                self.optimizer_D_PB = torch.optim.Adam(self.netD_PB.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.with_D_PP:
                self.optimizer_D_PP = torch.optim.Adam(self.netD_PP.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            if opt.with_D_PB:
                self.optimizers.append(self.optimizer_D_PB)
            if opt.with_D_PP:
                self.optimizers.append(self.optimizer_D_PP)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            if opt.with_D_PB:
                networks.print_network(self.netD_PB)
            if opt.with_D_PP:
                networks.print_network(self.netD_PP)
        print('-----------------------------------------------')

    def set_input(self, input):
        self.input_P1, self.input_BP1 = input['P1'], input['BP1']
        self.input_P2, self.input_BP2 = input['P2'], input['BP2']
        self.input_EP1, self.input_EP2 = input['EP1'], input['EP2']
        self.input_BEP2, self.input_BEP2c1 = input['BEP2'], input['BEP2c1']
        self.image_paths = input['P1_path'][0] + '___' + input['P2_path'][0]

        if len(self.gpu_ids) > 0:
            self.input_P1 = self.input_P1.cuda()
            self.input_BP1 = self.input_BP1.cuda()
            self.input_P2 = self.input_P2.cuda()
            self.input_BP2 = self.input_BP2.cuda()
            self.input_EP1 = self.input_EP1.cuda()
            self.input_EP2 = self.input_EP2.cuda()
            self.input_BEP2 = self.input_BEP2.cuda()
            self.input_BEP2c1 = self.input_BEP2c1.cuda()

    def forward(self):
        self.fake_ep2, self.fake_p2 = self.netG(self.input_P1, self.input_EP1, self.input_BP2, self.input_BEP2, self.input_EP2)

    def test(self):
        self.fake_ep2, self.fake_p2 = self.netG(self.input_P1, self.input_EP1, self.input_BP2, self.input_BEP2, self.input_EP2)


    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_person_paths(self):
        return self.person_paths


    def backward_G(self):
        if self.opt.with_D_PB:
            pred_fake_PB = self.netD_PB(torch.cat((self.fake_p2, self.input_BP2), 1))
            self.loss_G_GAN_PB = self.criterionGAN(pred_fake_PB, True)

        if self.opt.with_D_PP:
            pred_fake_PP = self.netD_PP(torch.cat((self.fake_p2, self.input_P1), 1))
            self.loss_G_GAN_PP = self.criterionGAN(pred_fake_PP, True)

        # CX loss
        cx_style_loss = 0
        if self.opt.use_cxloss:
            style_layer = ['r32', 'r42']
            vgg_style = self.vgg(self.input_P2, style_layer)
            vgg_fake = self.vgg(self.fake_p2, style_layer)
            cx_style_loss = 0

            for i, val in enumerate(vgg_fake):
                cx_style_loss += self.CX_loss(vgg_style[i], vgg_fake[i])
            cx_style_loss *= self.opt.lambda_cx

        pair_cxloss = cx_style_loss

        # L1 loss
        if self.opt.L1_type == 'l1_plus_perL1' :
            losses = self.criterionL1(self.fake_p2, self.input_P2)
            self.loss_G_L1 = losses[0]
            self.loss_originL1 = losses[1].data
            self.loss_perceptual = losses[2].data

        else:
            self.loss_G_L1 = self.criterionL1(self.fake_p2, self.input_P2) * self.opt.lambda_A

        pair_L1loss = self.loss_G_L1

        if self.opt.with_D_PB:
            pair_GANloss = self.loss_G_GAN_PB * self.opt.lambda_GAN
            if self.opt.with_D_PP:
                pair_GANloss += self.loss_G_GAN_PP * self.opt.lambda_GAN
                pair_GANloss = pair_GANloss / 2
        else:
            if self.opt.with_D_PP:
                pair_GANloss = self.loss_G_GAN_PP * self.opt.lambda_GAN


        if self.opt.with_D_PB or self.opt.with_D_PP:
            pair_loss = pair_L1loss + pair_GANloss
        else:
            pair_loss = pair_L1loss

        if self.opt.use_cxloss:
            pair_loss = pair_loss + pair_cxloss

        pair_loss.backward()

        self.pair_L1loss = pair_L1loss.data
        if self.opt.with_D_PB or self.opt.with_D_PP:
            self.pair_GANloss = pair_GANloss.data

        if self.opt.use_cxloss:
            self.pair_cxloss = pair_cxloss.data

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True) * self.opt.lambda_GAN
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False) * self.opt.lambda_GAN
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    # D: take(P, B) as input
    def backward_D_PB(self):
        real_PB = torch.cat((self.input_P2, self.input_BP2), 1)
        # fake_PB = self.fake_PB_pool.query(torch.cat((self.fake_p2, self.input_BP2), 1))
        fake_PB = self.fake_PB_pool.query( torch.cat((self.fake_p2, self.input_BP2), 1).data )
        loss_D_PB = self.backward_D_basic(self.netD_PB, real_PB, fake_PB)

        self.loss_D_PB = loss_D_PB.data

    # D: take(P, P') as input
    def backward_D_PP(self):
        real_PP = torch.cat((self.input_P2, self.input_P1), 1)
        # fake_PP = self.fake_PP_pool.query(torch.cat((self.fake_p2, self.input_P1), 1))
        fake_PP = self.fake_PP_pool.query( torch.cat((self.fake_p2, self.input_P1), 1).data )
        loss_D_PP = self.backward_D_basic(self.netD_PP, real_PP, fake_PP)

        self.loss_D_PP = loss_D_PP.data


    def optimize_parameters(self):
        # forward
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # D_P
        if self.opt.with_D_PP:
            for i in range(self.opt.DG_ratio):
                self.optimizer_D_PP.zero_grad()
                self.backward_D_PP()
                self.optimizer_D_PP.step()

        # D_BP
        if self.opt.with_D_PB:
            for i in range(self.opt.DG_ratio):
                self.optimizer_D_PB.zero_grad()
                self.backward_D_PB()
                self.optimizer_D_PB.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([ ('pair_L1loss', self.pair_L1loss)])
        if self.opt.with_D_PP:
            ret_errors['D_PP'] = self.loss_D_PP
        if self.opt.with_D_PB:
            ret_errors['D_PB'] = self.loss_D_PB
        if self.opt.with_D_PB or self.opt.with_D_PP or self.opt.with_D_PS:
            ret_errors['pair_GANloss'] = self.pair_GANloss

        if self.opt.L1_type == 'l1_plus_perL1':
            ret_errors['origin_L1'] = self.loss_originL1
            ret_errors['perceptual'] = self.loss_perceptual

        if self.opt.use_cxloss:
            ret_errors['CXLoss'] = self.pair_cxloss

        return ret_errors

    def get_current_visuals(self):
        height, width = self.input_P1.size(2), self.input_P1.size(3)
        input_P1 = util.tensor2im(self.input_P1.data)
        input_P2 = util.tensor2im(self.input_P2.data)
        input_EP1 = util.tensor2im(self.input_EP1.data)
        input_EP2 = util.tensor2im(self.input_EP2.data)

        input_BP1 = util.draw_pose_from_map(self.input_BP1.data)[0]
        input_BP2 = util.draw_pose_from_map(self.input_BP2.data)[0]

        fake_p2 = util.tensor2im(self.fake_p2.data)
        fake_ep2 = util.tensor2im(self.fake_ep2.data)

        vis = np.zeros((height, width*8, 3)).astype(np.uint8) #h, w, c
        vis[:, :width, :] = input_P1
        vis[:, width:width*2, :] = input_EP1
        vis[:, width*2:width*3, :] = input_BP1
        vis[:, width*3:width*4, :] = input_P2
        vis[:, width*4:width*5, :] = input_EP2
        vis[:, width*5:width*6, :] = input_BP2
        vis[:, width*6:width*7, :] = fake_ep2
        vis[:, width*7:, :] = fake_p2

        ret_visuals = OrderedDict([('vis', vis)])

        return ret_visuals

    def save(self, label, epoch, total_steps):
        self.save_network(self.netG,  'netG',  label, self.gpu_ids, epoch, total_steps)
        if self.opt.with_D_PB:
            self.save_network(self.netD_PB,  'netD_PB',  label, self.gpu_ids, epoch, total_steps)
        if self.opt.with_D_PP:
            self.save_network(self.netD_PP, 'netD_PP', label, self.gpu_ids, epoch, total_steps)
