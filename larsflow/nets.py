import torch
from torch import nn
from . import utils

import normflow as nf

class ConvNet2d(nn.Module):
    """
    Convolutional Neural Network with leaky ReLU nonlinearities and a
    final fully connected output layer
    """

    def __init__(self, channels, kernel_size, output_units, leaky=0.0,
                 output_fn=None):
        """
        Constructor
        :param channels: List of channels of conv layers, first entry is in_channels
        :param kernel_size: List of kernel sizes, same for height and width
        :param output_units: List of two ints
        :param leaky: Leaky part of ReLU
        :param output_fn: String, function to be applied to the output, either
        None, "sigmoid", or "clampexp"
        """
        super().__init__()
        # Build network
        net = nn.ModuleList([])
        for i in range(len(kernel_size)):
            net.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size[i],
                                 padding=kernel_size[i] // 2))
            net.append(nn.LeakyReLU(leaky))
        net.append(nn.Flatten())
        net.append(nn.Linear(*output_units))
        if output_fn == "sigmoid":
            net.append(nn.Sigmoid())
        elif output_fn == "clampexp":
            net.append(nf.utils.ClampExp())
        self.net = nn.Sequential(*net)

    def forward(self, x):
        if x.dim() == 3:
            x = x.view(-1, 1, x.size(1), x.size(2))
        return self.net(x)