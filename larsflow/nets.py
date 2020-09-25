import torch
from torch import nn
from . import utils

class ConvNet2d(nn.Module):
    """
    Convolutional Neural Network with leaky ReLU nonlinearities and a
    final fully connected output layer
    """

    def __init__(self, channels, kernel_size, output_units, leaky=0.0):
        """
        Constructor
        :param channels: List of channels of conv layers, first entry is in_channels
        :param kernel_size: List of kernel sizes, same for height and width
        :param output_units: List of two ints
        :param leaky: Leaky part of ReLU
        """
        super().__init__()
        # Build network
        net = nn.ModuleList([])
        for i in range(len(kernel_size)):
            net.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size[i],
                                 padding=kernel_size[i] // 2))
            net.append(nn.LeakyReLU(leaky))
        net.append(nn.Flatten())
        net.append(nn.Linear(output_units[0], output_units[0]))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        if x.dim() == 3:
            x = x.view(-1, 1, x.size(1), x.size(2))
        return self.net(x)