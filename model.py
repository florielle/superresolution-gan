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

lr_dir = "/scratch/jmw784/superresolution/data/lr/"
hr_dir = "/scratch/jmw784/superresolution/data/hr/"
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = srData(lr_dir, hr_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

print(dataloader.dataset[0])

