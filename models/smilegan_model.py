import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from .base_model import BaseModel
from . import networks


class SmileGANModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_D', type=float, default=0.8, help='weight for local Discriminator loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.input_size = opt.crop_size
        self.anchor = opt.anchor
        self.num_label = opt.num_label
        self.G_inp_channel = 3
        self.D_inp_channel = 3
        self.num_frames = opt.num_frames
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'G_mask', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'Gm', 'D']

        else:  # during test time, only load G
            self.model_names = ['G', 'Gm']

        self.netG = networks.define_G(self.G_inp_channel, 3 * (opt.num_frames - 1), opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netGm = networks.define_G(self.G_inp_channel, 3 * (opt.num_frames - 1), opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain or self.isFineTune: # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc

            self.D_inp_channel += 3 * (opt.num_frames - 1)
            self.netD = networks.define_D(self.D_inp_channel, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionGDL = networks.gdl_loss
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            G_params = list(self.netG.parameters()) + list(self.netGm.parameters())
            D_params = list(self.netD.parameters())

            self.optimizer_G = torch.optim.Adam(G_params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(D_params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        if self.isFineTune:
            self.criterionGAN = networks.GANLoss('lsgan').to(self.device)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        in_tensor = self.real_A

        self.foreground = self.netG(in_tensor)  # G(A)
        self.mask = self.netGm(in_tensor)
        inv_mask = torch.ones(self.mask.size()).to(self.device)-self.mask

        self.background = self.real_A.repeat(1, self.num_frames - 1, 1, 1)

        self.fake_B = self.mask*self.foreground + inv_mask*self.background
        # self.fake_B = self.foreground
        return self.fake_B

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        # print(self.fake_B.size())
        pred_fake = self.netD(fake_AB.detach())

        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        # print(self.real_A.size(),self.real_B.size())
        self.real_B = self.real_B.view(1, 3 * (self.num_frames - 1), self.input_size, self.input_size)
        real_AB = torch.cat((self.real_A, self.real_B), 1)

        pred_real = self.netD(real_AB)

        self.loss_D_real =self.criterionGAN(pred_real, True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)

        pred_fake = self.netD(fake_AB)

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G_mask =  0 * self.criterionL1(self.mask, torch.zeros(self.mask.size()).to(self.device))
        # self.loss_G_GDL = self.criterionGDL(self.fake_B, self.real_B) * 50
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_mask
        self.loss_G.backward()


    def optimize_parameters(self):
        fb = self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
