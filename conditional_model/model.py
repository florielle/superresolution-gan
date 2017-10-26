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



# discriminator
# fix architecture ***
# change input to be concatnated images

class net_d(nn.Module):
    """
    Discriminator
    """
    def __init__(self):
        super(net_d, self).__init__()
        self.relu = nn.ReLU()
        self.conv5 = nn.Conv2d(6, 32, (5, 5), (1, 1), (2, 2))
        self.conv6 = nn.Conv2d(32, 3, (3, 3), (1, 1), (1, 1))
        self.lin1 = nn.Linear(256, 1)

        
    def forward(self, data):
        x = self.relu(self.conv5(data))
        x = self.relu(self.conv6(x))
        x = self.lin1(x)
        return nn.functional.sigmoid(x.view((32, -1)))

    
    
