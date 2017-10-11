import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def name_list(directory):
    names = []
    
    for filename in os.listdir(directory): 
        if filename.endswith(".jpg"):
            names.append(filename)
    return names

class srData(data.Dataset):
    def __init__(self, lr_root, hr_root, transform=None):
        self.lr_root = lr_root
        self.hr_root = hr_root
        self.names = name_list(lr_root)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        filename = self.names[index]
        lr_img = pil_loader(self.lr_root+filename)
        hr_img = pil_loader(self.hr_root+filename)

        if self.transform is not None:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)

        return lr_img, hr_img

    def __len__(self):
        return len(self.names)
