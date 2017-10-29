### epoch loops
import torch
import torch.backends.cudnn as cudnn

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
import numpy as np
from dataloader import srData, pil_loader, name_list
import random
from pdb import set_trace as st
from model import net_g, net_d

cudnn.benchmark = True

lr_dir = "/scratch/mmd378/DSGA1013/project/data/lr/"
hr_dir = "/scratch/mmd378/DSGA1013/project/data/hr/"
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = srData(lr_dir, hr_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)




# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
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

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

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
        for image in images.data:
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
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images


if __name__ == '__main__':
    l_rate = 0.001
    G = net_g()
    D = net_d()
    net_d.cuda()
    net_g.cuda()

    beta_1 = 0.5
    criterionGAN = GANLoss()
    criterionL1 = torch.nn.L1Loss()
    optimizer_G = torch.optim.Adam(G.parameters(),
                                        lr=l_rate, betas=(beta_1, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(),
                                        lr=l_rate, betas=(beta_1, 0.999))

    fake_lrhr_pool = ImagePool(3)
    lambda_A = 0.5

    for epoch in range(5):
        batch_num = 0
        for batch in dataloader:
            batch_num +=1
            real_lr_small, real_hr = Variable(batch[0]), Variable(batch[1])
            #torch.Size([32, 3, 64, 64]) torch.Size([32, 3, 256, 256])
            # forward step
            fake_hr = G.forward(real_lr_small)


            # get padded version of real_lr
            padding = Variable(torch.zeros(32, 3, 192, 64))
            padding2 = Variable(torch.zeros(32, 3, 256, 192))
            real_lr = torch.cat((torch.cat((real_lr_small, padding), 2), padding2), 3)

            # Optimize step 
            optimizer_D.zero_grad()

            # backward_D()
            fake_lrhr = fake_lrhr_pool.query(torch.cat((real_lr, fake_hr), 1))
            pred_fake = D.forward(fake_lrhr.detach())
            loss_D_fake = criterionGAN(pred_fake, False)

            # Real
            real_lrhr = torch.cat((real_lr, real_hr), 1)#.detach()
            pred_real = D.forward(real_lrhr)
            loss_D_real = criterionGAN(pred_real, True)

            # Combined loss
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            # end backward_D()

            loss_D.backward()



            optimizer_D.step()

            optimizer_G.zero_grad()
            # backward_G()
            fake_AB = torch.cat((real_lr, fake_hr), 1)
            pred_fake = D.forward(fake_AB)
            loss_G_GAN = criterionGAN(pred_fake, True)

            # Second, G(A) = B
            loss_G_L1 = criterionL1(fake_hr, real_hr) * lambda_A

            loss_G = loss_G_GAN + loss_G_L1

            loss_G.backward()

            optimizer_G.step()

    torch.save(G.cpu().state_dict(), "G_save.pth")
    torch.save(D.cpu().state_dict(), "D_save.pth")



            
            
    
    
