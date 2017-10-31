import torch
import torch.nn as nn
import sys

sys.dont_write_bytecode = True

class SRGAN_D(nn.Module):
    def __init__(self, nc, ngpu, bn):
        super(SRGAN_D, self).__init__()
        self.bn = bn
        self.ngpu = ngpu
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.conv1 = nn.Conv2d(nc, 64, 4, 2, 1, bias=True)
        self.conv2 = nn.Conv2d(64, 32, 4, 2, 1, bias=True)
        self.conv3 = nn.Conv2d(32, 16, 4, 2, 1, bias=True)
        self.conv4 = nn.Conv2d(16, 8, 4, 2, 1, bias=True)
        
        if self.bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn3 = nn.BatchNorm2d(16)
            self.bn4 = nn.BatchNorm2d(8)

        self.linear = nn.Linear(8*16*16, 1)

    def forward(self, x):
        if self.bn:
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.dropout(self.relu(self.bn2(self.conv2(x))))
            x = self.dropout(self.relu(self.bn3(self.conv3(x))))
            x = self.relu(self.bn4(self.conv4(x)))
        else:
            x = self.relu(self.conv1(x))
            x = self.dropout(self.relu(self.conv2(x)))
            x = self.dropout(self.relu(self.conv3(x)))
            x = self.relu(self.conv4(x))

        x = x.view(x.size(0), -1)
        x = self.linear(x)
        
        return x

class SRGAN_G(nn.Module):
    def __init__(self, nc, ngpu, bn):
        super(SRGAN_G, self).__init__()
        self.bn = bn
        self.ngpu = ngpu
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.conv1 = nn.Conv2d(nc, 64, 5, 1, 2, bias=True)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(32, 3*4 ** 2, 3, 1, 1, bias=True)

        if self.bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(32)

        self.pixel_shuffle = nn.PixelShuffle(4)
        self.tanh = nn.Tanh()

    def forward(self, x):

        if self.bn:
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.dropout(self.relu(self.bn2(self.conv2(x))))
            x = self.dropout(self.relu(self.bn3(self.conv3(x))))
        
        else:
            x = self.relu(self.conv1(x))
            x = self.dropout(self.relu(self.conv2(x)))
            x = self.dropout(self.relu(self.conv3(x)))

        x = self.pixel_shuffle(self.conv4(x))
        x = self.tanh(x)
        return x
