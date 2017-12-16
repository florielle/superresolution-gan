### epoch loops
import torch
import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as data
from torch.autograd import Variable
from PIL import Image
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse

import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from comet_ml import Experiment


import numpy as np
from dataloader import srData, pil_loader, name_list

import random
from pdb import set_trace as st

from model import net_g, net_d_lite_2_conv_3 as net_d, SRGAN_D, init_model, CGAN_D, DCGAN_D, net_g_dropout

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
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
parser.add_argument('--loss', type=str, default='mse', help='mse or ganloss loss')
parser.add_argument('--output_path', type=str, default='output', help='path identifier')
parser.add_argument('--beta_1', type=float, default=0.5, help='beta1 for adam optim')
parser.add_argument('--oldmodelD', action='store_true', help='use SRGAN D or previous')
parser.add_argument('--CGAND', action='store_true', help='use CGAND')
parser.add_argument('--D_pretrain', type=int, default=100, help='Pretrain D')
parser.add_argument('--eps', type=float, default=1e-12, help='epsilon')
parser.add_argument('--num_channels', type=int, default=3)
parser.add_argument('--slope', type=float, default=0.2, help='for leaky ReLU')
parser.add_argument('--std', type=float, default=0.02, help='for weight')
parser.add_argument('--wasserstein', action='store_true')
parser.add_argument('--clamp', type=float, default=1e-2)
parser.add_argument('--total_epochs', type=int, default=15000)
parser.add_argument('--netg_dropout', action='store_true')



opt = parser.parse_args()
print(opt)

experiment = Experiment(api_key="aSOO6NAHZQ5CPwhIJWrjPuPdf", log_code=True)
experiment.log_multiple_params(vars(opt))





lr_dir = "/scratch/mmd378/DSGA1013/project/data/lr/"
hr_dir = "/scratch/mmd378/DSGA1013/project/data/hr/"
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = srData(lr_dir, hr_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)




# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, inpt, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != inpt.numel()))
            if create_label:
                real_tensor = self.Tensor(inpt.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != inpt.numel()))
            if create_label:
                fake_tensor = self.Tensor(inpt.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, inpt, target_is_real):
        target_tensor = self.get_target_tensor(inpt, target_is_real)
        return self.loss(inpt, target_tensor)

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images: #.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image.cuda())
        return_images = Variable(torch.cat(return_images, 0).cuda(), volatile=True)
        return return_images

def updated_training_loop():

    gen_iterations = 0
    if opt.oldmodelD:
        D = net_d()
        init_model(D, 'normal')
    else:
        D=SRGAN_D(6,1)
        init_model(D, 'normal')
    if opt.CGAND:
        D = CGAN_D(opt)
        init_model(D, 'normal')
    if opt.wasserstein:
        D=DCGAN_D(256,6,32,1)
    if opt.netg_dropout:
        G=net_g_dropout()
    else:
        G = net_g()
    #init_model(G, 'normal')
    #
    D.train()
    G.train()
    losses = []
    start = time.time()
    EPS = opt.eps
    G.cuda()
    D.cuda()
    beta_1 = 0.5
    if opt.loss == 'mse':
        criterionGAN = nn.MSELoss() #GANLoss()
    elif opt.loss =='ganloss':
        criterionGAN = GANLoss()

    criterionL1 = torch.nn.L1Loss()
    if opt.wasserstein:
        optimizer_D = torch.optim.RMSprop(D.parameters(), lr=opt.lrD)
        optimizer_G = torch.optim.RMSprop(G.parameters(), lr=opt.lrG)
    else:
        optimizer_G = torch.optim.Adam(G.parameters(),
                                            lr=opt.lrG, betas=(opt.beta_1, 0.999))
        optimizer_D = torch.optim.Adam(D.parameters(),
                                            lr=opt.lrD, betas=(opt.beta_1, 0.999))

    total_epochs = opt.total_epochs
    for epoch in range(total_epochs):
        #print(epoch)
        experiment.log_current_epoch(epoch)

        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):

            ############################
            # (1) Update D network
            ###########################
            for p in D.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            # train the discriminator Diters times
            if epoch < opt.D_pretrain or (epoch+1) % 50 == 0:
                Diters = 100
            else:
                Diters = opt.Diters
            j = 0
            #Diters = 5
            while j < Diters and i < len(dataloader):
                j += 1

                lr, hr = data_iter.next()
                i += 1

                # Drop the last batch if it's not the same size as the batchsize
                if lr.size(0) != opt.batch_size:
                    break
                lr = lr.cuda()
                hr = hr.cuda()

                # train with real

                D.zero_grad()

                padding = torch.zeros(opt.batch_size, 3, 192, 64).cuda()
                padding2 = torch.zeros(opt.batch_size, 3, 256, 192).cuda()
                real_lr = torch.cat((torch.cat((lr, padding), 2), padding2), 3)

                inputy = Variable(torch.cat((real_lr, hr), 1))
                #inputy = Variable(hr)

                errD_real = D(inputy, True) # can modify to feed inputv too

                padding = torch.zeros(opt.batch_size, 3, 192, 64).cuda()
                padding2 = torch.zeros(opt.batch_size, 3, 256, 192).cuda()
                real_lr = torch.cat((torch.cat((lr, padding), 2), padding2), 3)


                # completely freeze netG while we train the discriminator
                inputg = Variable(lr, volatile = True)
                #fake = Variable(netG(inputg).data)
                fake = Variable(torch.cat((real_lr, G(inputg).data), 1))
                errD_fake = D(fake, True)

                # calculate discriminator loss and backprop
                # errD = 0.5 * (torch.mean((errD_real - 1)**2) + torch.mean(errD_fake**2))
                if opt.wasserstein:
                    errD = -torch.mean(errD_real)+torch.mean(errD_fake)
                else:
                    errD = -torch.mean(torch.log(errD_real+EPS) +torch.log(1-errD_fake+EPS))

                errD.backward()

                optimizer_D.step()
                if opt.wasserstein:
                    for p in D.parameters():
                        p.data.clamp_(-opt.clamp, opt.clamp)

    #         print("here")
            ############################
            # (2) Update G network
            ###########################
            for p in D.parameters():
                p.requires_grad = False # to avoid computation
            G.zero_grad()

            lr = lr.cuda()
            hr = hr.cuda()
            if lr.size(0) != opt.batch_size:
                break
            input_lr = Variable(lr)
            input_hr = Variable(hr)        

            padding = torch.zeros(opt.batch_size, 3, 192, 64).cuda()
            padding2 = torch.zeros(opt.batch_size, 3, 256, 192).cuda()
    #         print(lr.size())
    #         print(padding.size())
    #         print(padding2.size())
            real_lr = torch.cat((torch.cat((lr, padding), 2), padding2), 3)

            fake = G(input_lr)

            #errG_1 = D(Variable(torch.cat((real_lr, fake.data), 1),requires_grad=True))
            errG_1 = D(torch.cat((Variable(real_lr), fake), 1))
            # maximize log(D) instead of minimize log(1 - D)
            if opt.wasserstein:
                errG = -torch.mean(errG_1)
            else:
                errG = -torch.mean(torch.log(errG_1+EPS))
            # generator accumulates loss from discriminator + MSE with true image
            # loss_MSE = MSE(fake, input_hr)
            # loss_SSIM = ssim_loss(fake, input_hr)
            # loss_G = 0.25*errG + loss_MSE - loss_SSIM
            loss_G = errG
            #loss_G = criterionL1(fake, input_hr)
            loss_G.backward()
            losses.append(loss_G.data)
            optimizer_G.step()
            gen_iterations += 1


        if epoch > opt.D_pretrain and (epoch+1) % 1 == 0:
            print_loss_avg = np.mean(losses).cpu().numpy()[0]
            experiment.log_metric("average_g_loss",100 * print_loss_avg)
            experiment.log_metric("actual_g_loss", losses[-1].cpu().numpy()[0] * 100)
            print('%s (%d %d%%) %.4f' % (timeSince(start, (epoch+1) / total_epochs),
                                             epoch, epoch / total_epochs * 100, print_loss_avg))

        if (epoch+1) % 100 == 0:
            print('saving model at epoch', epoch)
            torch.save(G.state_dict(), "checkpoints/G_{0}_{1}.pth".format(opt.output_path, epoch))
            torch.save(D.state_dict(), "checkpoints/D_{0}_{1}.pth".format(opt.output_path, epoch))


if __name__ == '__main__':
    updated_training_loop()
    
