import torch
import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
# import util.util as util
# from util.image_pool import ImagePool
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as data
from torch.autograd import Variable
from PIL import Image
import numpy as np
from dataloader import srData, pil_loader, name_list
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.nn.functional as F

# generator

class net_g(nn.Module):
    """
    Generator
    """
    # take G from: https://github.com/pytorch/examples/blob/master/super_resolution/model.py
    def __init__(self, upscale_factor=4):
        super(net_g, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, 3*upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        nn.init.orthogonal(self.conv1.weight, nn.init.calculate_gain('relu'))
        nn.init.orthogonal(self.conv2.weight, nn.init.calculate_gain('relu'))
        nn.init.orthogonal(self.conv3.weight, nn.init.calculate_gain('relu'))
        nn.init.orthogonal(self.conv4.weight)

class net_g_dropout(nn.Module):
    """
    Generator
    """
    # take G from: https://github.com/pytorch/examples/blob/master/super_resolution/model.py
    def __init__(self, upscale_factor=4):
        super(net_g_dropout, self).__init__()
        self.relu = nn.ReLU()
        self.dropout=nn.Dropout2d(p=0.5)
        self.conv1 = nn.Conv2d(3, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, 3*upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        nn.init.orthogonal(self.conv1.weight, nn.init.calculate_gain('relu'))
        nn.init.orthogonal(self.conv2.weight, nn.init.calculate_gain('relu'))
        nn.init.orthogonal(self.conv3.weight, nn.init.calculate_gain('relu'))
        nn.init.orthogonal(self.conv4.weight)


# discriminator
# fix architecture ***

class net_d(nn.Module):
    """
    Discriminator
    """
    def __init__(self):
        super(net_d, self).__init__()
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout2d(p=0.9)
#         self.conv5 = nn.Conv2d(6, 32, (5, 5), (1, 1), (2, 2))
#         self.conv6 = nn.Conv2d(32, 3, (3, 3), (1, 1), (1, 1))
#         self.lin1 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.5)
        self.conv5 = nn.Conv1d(2000, 512, 2)
        self.conv6 = nn.Conv1d(256, 32, 2)
        self.lin0 = nn.Linear(196608, 2000)
        self.lin2 = nn.Linear(32, 1)
        self.batchnorm=nn.BatchNorm2d(6)
        
    def forward(self, data, batchnorm=False):
#         x = data
#         x = self.dropout(x)
#         x = self.relu(self.conv5(x))
#         x = self.relu(self.conv6(x))
#         x = self.lin1(x)
        if batchnorm:
            x = self.batchnorm(data)
        else:
            x = data
        x = x.view(32,2,-1)
        x = self.lin0(x)
        x = self.dropout(x)
        x = x.view(32, 2000, -1)
        x = self.conv5(x)
        x = self.relu(self.conv6(x.view(32, 256, -1)))
        x = self.lin2(x.view((32, -1)))
        return nn.functional.sigmoid(x.view((32, -1)))


class net_d_lite(nn.Module):
    """
    Discriminator
    """
    def __init__(self):
        super(net_d_lite, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.5)
        self.lin3 = nn.Linear(6*256*256,1)
        self.batchnorm=nn.BatchNorm2d(6)
        
    def forward(self, data, batchnorm=False):
        if batchnorm:
            x = self.batchnorm(data)
        else:
            x = data
        x = x.view(32,-1)
        x = self.dropout(x)
        x = self.lin3(x)
        return nn.functional.sigmoid(x.view((32, -1)))

    
class net_d_lite_conv(nn.Module):
    """
    Discriminator
    """
    def __init__(self):
        super(net_d_lite_conv, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.5)
        self.lin3 = nn.Linear(127008,1)
        self.conv1 = nn.Conv2d(6,2,5)
        self.batchnorm=nn.BatchNorm2d(6)
        
    def forward(self, data, batchnorm=False):
        if batchnorm:
            x = self.batchnorm(data)
        else:
            x = data
        x = self.relu(self.conv1(x))

        x = x.view(32,-1)
        x = self.dropout(x)
        x = self.lin3(x)
        return nn.functional.sigmoid(x.view((32, -1)))

class net_d_lite_2_conv(nn.Module):
    """
    Discriminator
    """
    def __init__(self):
        super(net_d_lite_2_conv, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.5)
        self.lin3 = nn.Linear(127008,1)
        self.conv1 = nn.Conv2d(6,2,5)
        self.batchnorm=nn.BatchNorm2d(6)
        self.lin3 = nn.Linear(123008,1)
        self.conv1 = nn.Conv2d(6,2,5)
        self.conv2 = nn.Conv2d(2,2,(5,5))

        
    def forward(self, data, batchnorm=False):
        if batchnorm:
            x = self.batchnorm(data)
        else:
            x = data
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = x.view(32,-1)
        x = self.lin3(x)
        return nn.functional.sigmoid(x.view((32, -1)))
    
class net_d_lite_2_conv_1(nn.Module):
    """
    Discriminator
    """
    def __init__(self):
        super(net_d_lite_2_conv_1, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.1)
        self.batchnorm=nn.BatchNorm2d(6)
        self.lin3 = nn.Linear(12800,1)
        self.conv1 = nn.Conv2d(6,3,5)
        self.conv2 = nn.Conv2d(3,2,(5,5))
        self.conv3 = nn.Conv2d(3,3,(3,3), (3,3))


        
    def forward(self, data, batchnorm=False):
        if batchnorm:
            x = self.batchnorm(data)
        else:
            x = data
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv2(x))
        x = x.view(32,-1)
        x = self.lin3(x)
        return nn.functional.sigmoid(x.view((32, -1)))


class net_d_lite_2_conv_2(nn.Module):
    """
    Discriminator
    """
    def __init__(self):
        super(net_d_lite_2_conv_2, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.1)
        self.batchnorm=nn.BatchNorm2d(6)
        self.lin3 = nn.Linear(12800,1)
        self.conv1 = nn.Conv2d(6,3,5)
        self.conv2 = nn.Conv2d(3,2,(5,5))
        self.conv3 = nn.Conv2d(3,3,(3,3), (3,3))


        
    def forward(self, data, batchnorm=False):
        if batchnorm:
            x = self.batchnorm(data)
        else:
            x = data
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.conv2(x)
        x = x.view(32,-1)
        x = self.lin3(x)
        return nn.functional.sigmoid(x.view((32, -1)))
    
    
class net_d_lite_2_conv_3(nn.Module):
    """
    Discriminator
    """
    def __init__(self):
        super(net_d_lite_2_conv_3, self).__init__()
        self.batchnorm=nn.BatchNorm2d(6)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.9)
        self.conv5 = nn.Conv2d(6, 32, (5, 5), (1, 1), (2, 2))
        self.conv6 = nn.Conv2d(32, 3, (3, 3), (1, 1), (1, 1))
        self.lin1 = nn.Linear(256, 1)
        self.lin2 = nn.Linear(768, 1)


        
    def forward(self, data, batchnorm=False):
        if batchnorm:
            x = self.batchnorm(data)
        else:
            x = data
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.lin1(x)
        x = self.lin2(x.view(32,-1))
        return nn.functional.sigmoid(x.view((32, -1)))


# Custom weights initialization called on netG and netD
#if opt.init not in ['normal', 'xavier', 'kaiming']:
#    print('Initialization method not found, defaulting to normal')

def init_model(model, init):
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            if init == 'xavier':
                m.weight.data = init.xavier_normal(m.weight.data)
            elif init == 'kaiming':
                m.weight.data = init.kaiming_normal(m.weight.data)
            else:
                m.weight.data.normal_(0.0, 0.02)
            
            m.bias.data.fill_(0)

        elif isinstance(m,nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

"""
Define model
"""

# Return the correct nonlinearity based on argument
def which_relu(nonlinearity='relu'):
    if nonlinearity == 'selu':
        return nn.SELU()
    elif nonlinearity == 'prelu':
        return nn.PReLU()
    elif nonlinearity == 'leaky':
        return nn.LeakyReLU()
    else:
        return nn.ReLU()

# Basic conv -> bn -> relu -> pool -> dropout module
class BasicConv2d_D(nn.Module):

    def __init__(self, in_channels, out_channels,dropout=0.5, **kwargs):
        super(BasicConv2d_D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = which_relu()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        return x

# Basic conv -> bn -> relu -> dropout module
class BasicConv2d_G(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=0.5, resizeConv=False,  **kwargs):
        super(BasicConv2d_G, self).__init__()
        self.resizeConv=resizeConv
        if self.resizeConv:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = which_relu()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        if self.resizeConv:
            x = F.avg_pool2d(self.conv(self.upsample(x)),2)
        else:
            x = self.conv(x)

        x = self.relu(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

class SRGAN_D(nn.Module):
    def __init__(self, nc, ngpu):
        super(SRGAN_D, self).__init__()
        self.ngpu = ngpu
        self.conv1 = BasicConv2d_D(nc, 32, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2 = BasicConv2d_D(32, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = BasicConv2d_D(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = BasicConv2d_D(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = BasicConv2d_D(64, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.linear = nn.Linear(2048, 1)

    def forward(self, x, bn_placeholder=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class SRGAN_G(nn.Module):
    def __init__(self, nc, ngpu):
        super(SRGAN_G, self).__init__()
        self.ngpu = ngpu
        self.conv1 = BasicConv2d_G(nc, 32, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2 = BasicConv2d_G(32, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = BasicConv2d_G(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = BasicConv2d_G(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = BasicConv2d_G(64, 3 * 4**2, kernel_size=3, stride=1, padding=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(4)
        self.tanh = nn.Tanh()

    def forward(self, x, bn_placeholder=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pixel_shuffle(x)
        x = self.tanh(x)
        return x

# https://github.com/ozansener/GAN/blob/master/cgan/model.py
class CGAN_D(nn.Module):
    def __init__(self, opt, ndf=64):
        super(CGAN_D, self).__init__()
        self.opt = opt
        self.conv1 = nn.Conv2d(opt.num_channels*2, ndf, 4, stride=2, padding=1)
        self.lr = nn.LeakyReLU(opt.slope, inplace=True)
        
        # state dim: ndf x 128 x 128
        self.conv2 = nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(ndf*2)
        
        # state dim: ndf*2 x 64 x 64
        self.conv3=nn.Conv2d(ndf*2, ndf*4, 4, stride=2, padding=1)
        #self.bn2=nn.BatchNorm2d(ndf*2)
        self.bn2=nn.BatchNorm2d(256)
        
        # state dim: ndf*4 x 32 x 32
        self.conv4=nn.Conv2d(ndf*4, ndf*8, 4, stride=1, padding=1)
        self.bn3=nn.BatchNorm2d(ndf*4)
        
        # state dim: ndf*8 x 31 x 31
        self.conv5 = nn.Conv2d(ndf*8, 1, 4, stride=1, padding=1)
        self.sig = nn.Sigmoid()
        # output dim: ndf*8 x 30 x 30

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, self.opt.std)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, self.opt.std)
                m.bias.data.zero_()

    def forward(self, x,placeholder=True):
        x = self.conv1(x)
        x = self.lr(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = self.lr(x)
        
        x = self.conv3(x)
        #x = self.bn2(x)
        x = self.lr(x)
        
        x = self.conv4(x)
        #x = self.bn3(x)
        x = self.lr(x)

        
        x = self.conv5(x)
        return self.sig(x)

    
    
class DCGAN_D(nn.Module):
    def __init__(self, isize, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial.conv.{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial.relu.{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid.{0}-{1}.conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid.{0}.relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final.{0}-{1}.conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main


    def forward(self, input,placeholder=True):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else: 
            output = self.main(input)
            
        output = output.mean(0)
        return output.view(1)
    
