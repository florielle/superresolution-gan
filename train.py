import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.init as init
from torch.autograd import Variable
import os
import numpy as np
from PIL import Image
from dataloader import *
from comet_ml import Experiment
import pytorch_ssim

"""
Options for training
"""

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--lrSize', type=int, default=64, help='the height / width of the low resolution image')
parser.add_argument('--hrSize', type=int, default=256, help='the height / width of the high resolution image')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--niter', type=int, default=4000, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--init', type=str, default='normal', help='initialization method (normal, xavier, kaiming)')
parser.add_argument('--Dweight', type=float, default=1.0, help='weighting for G loss from D') 
parser.add_argument('--nonlinearity', type=str, default='relu', help='Nonlinearity to use (selu, prelu, leaky, relu)')
parser.add_argument('--resizeConv', action='store_true', help='Whether to use resize convolution layers in the generator')
parser.add_argument('--dropout', type=float, default=0.5, help='Percent dropout, default = 0.5')
opt = parser.parse_args()
print(opt)

experiment = Experiment(api_key="lN9B6VboZhaXa6TkJcdAZAfSf", log_code=True)
hyper_params = vars(opt)
experiment.log_multiple_params(hyper_params)

"""
Load experiments 
"""

if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))
os.system('mkdir {0}/images'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

###############################################################################
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

lr_dir = "/scratch/jmw784/superresolution/data/lr/"
hr_dir = "/scratch/jmw784/superresolution/data/hr/"
dataset = srData(lr_dir, hr_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True)

ngpu = int(opt.ngpu)
nc = int(opt.nc)

# Custom weights initialization called on netG and netD
if opt.init not in ['normal', 'xavier', 'kaiming']:
    print('Initialization method not found, defaulting to normal')

def init_model(model):
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            if opt.init == 'xavier':
                m.weight.data = init.xavier_normal(m.weight.data)
            elif opt.init == 'kaiming':
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
def which_relu():
    if opt.nonlinearity == 'selu':
        return nn.SELU()
    elif opt.nonlinearity == 'prelu':
        return nn.PReLU()
    elif opt.nonlinearity == 'leaky':
        return nn.LeakyReLU()
    else:
        return nn.ReLU()

# Basic conv -> bn -> relu -> pool -> dropout module
class BasicConv2d_D(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d_D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = which_relu()
        self.dropout = nn.Dropout(p=opt.dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        return x

# Basic conv -> bn -> relu -> dropout module
class BasicConv2d_G(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d_G, self).__init__()
        
        if opt.resizeConv:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = which_relu()
        self.dropout = nn.Dropout(opt.dropout)

    def forward(self, x):

        if opt.resizeConv:
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

    def forward(self, x):
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pixel_shuffle(x)
        x = self.tanh(x)
        return x

# Create model objects
netG = SRGAN_G(nc, ngpu)
init_model(netG)
netG.train()

netD = SRGAN_D(nc, ngpu)
init_model(netD)
netD.train()

# Define MSE loss module
MSE = nn.MSELoss()

# Define SSIM loss module
ssim_loss = pytorch_ssim.SSIM(window_size = 11)

# Load checkpoint models if needed
if opt.netG != '': 
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

if opt.cuda:
    netD.cuda()
    netG.cuda()

# Set up optimizer
if opt.adam:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

# Training loop
gen_iterations = 0

for epoch in range(opt.niter+1):
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader):

        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        # train the discriminator Diters times
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
        else:
            Diters = opt.Diters
        j = 0

        while j < Diters and i < len(dataloader):
            j += 1

            lr, hr = data_iter.next()
            i += 1

            # Drop the last batch if it's not the same size as the batchsize
            if lr.size(0) != opt.batchSize:
                break

            if opt.cuda:
                lr = lr.cuda()
                hr = hr.cuda()

            # train with real

            netD.zero_grad()

            inputy = Variable(hr)

            errD_real = netD(inputy) # can modify to feed inputv too

            # completely freeze netG while we train the discriminator
            inputg = Variable(lr, volatile = True)
            fake = Variable(netG(inputg).data)

            errD_fake = netD(fake)

            # calculate discriminator loss and backprop
            errD = 0.5 * (torch.mean((errD_real - 1)**2) + torch.mean(errD_fake**2))
            errD.backward()
                                        
            optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation
        netG.zero_grad()

        if opt.cuda:
            lr = lr.cuda()
            hr = hr.cuda()

        input_lr = Variable(lr)
        input_hr = Variable(hr)        

        fake = netG(input_lr)
        errG_1 = netD(fake)
        errG = 0.5 * torch.mean((errG_1 - 1)**2)

        # generator accumulates loss from discriminator + MSE with true image
        loss_MSE = MSE(fake, input_hr)
        loss_SSIM = ssim_loss(fake, input_hr)
        loss_G = opt.Dweight*errG + loss_MSE - loss_SSIM
        loss_G.backward()

        optimizerG.step()
        gen_iterations += 1

        print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f, MSE_Loss: %f'
            % (epoch, opt.niter, i, len(dataloader), gen_iterations,
            errD.data[0], loss_G.data[0], loss_MSE.data[0]))
        if gen_iterations % 500 == 0:
            vutils.save_image(hr, '{0}/images/{1}_real.png'.format(opt.experiment, gen_iterations), normalize=True)
            netG.eval()
            fake = netG(Variable(lr, volatile=True))
            netG.train()
            vutils.save_image(fake.data, '{0}/images/{1}_fake.png'.format(opt.experiment, gen_iterations), normalize=True)

    # Send to comet only once per epoch
    experiment.log_metric("MSE loss", loss_MSE.data[0])
    experiment.log_metric("SSIM loss", loss_SSIM.data[0])
    experiment.log_metric("Generator loss", errG.data[0])
    experiment.log_metric("Discriminator loss", errD.data[0])


    # do checkpointing
    if epoch % 100 == 0:
        torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
        torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))
