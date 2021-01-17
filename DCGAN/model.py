import torch
from torch.nn.modules.conv import ConvTranspose1d
import torch.nn.parallel
import torch.nn as nn
import torch.optim as optim


class ConvTranspose2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, input):
        return self.model(input)


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.model(input)


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels, out_channels, h = config['in_channels'], config['out_channels'], config['hidden_dims']
        self.model = nn.Sequential(
            ConvTranspose2dBlock(in_channels, h[0], 4, 1, 0),
            ConvTranspose2dBlock(h[0], h[1]),
            ConvTranspose2dBlock(h[1], h[2]),
            ConvTranspose2dBlock(h[2], h[3]),
            nn.ConvTranspose2d(h[3], out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels, out_channels, h = config['in_channels'], config['out_channels'], config['hidden_dims']
        self.model = nn.Sequential(
            Conv2dBlock(out_channels, h[0]),
            Conv2dBlock(h[0], h[1]),
            Conv2dBlock(h[1], h[2]),
            Conv2dBlock(h[2], h[3]),
            nn.Conv2d(h[3], in_channels, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)
