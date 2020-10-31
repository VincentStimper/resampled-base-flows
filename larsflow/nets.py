import torch
from torch import nn
from . import utils

import normflow as nf

class ConvNet2d(nn.Module):
    """
    Convolutional Neural Network with leaky ReLU nonlinearities and a
    final fully connected output layer
    """

    def __init__(self, channels, output_units, kernel_size=3, stride=1,
                 leaky=0.0, output_fn=None):
        """
        Constructor
        :param channels: List of channels of conv layers, first entry is in_channels
        :param kernel_size: Int of list of ints, same for height and width, if int
        same kernel size for each layer is chosen
        :param output_units: List of two ints
        :param stride: Int or list of int, if int same stride for all layers is used
        :param leaky: Leaky part of ReLU
        :param output_fn: String, function to be applied to the output, either
        None, "sigmoid", or "clampexp"
        """
        super().__init__()
        # Prepare parameters
        n_layers = len(channels) - 1
        if isinstance(stride, int):
            stride = n_layers * [stride]
        if isinstance(kernel_size, int):
            kernel_size = n_layers * [kernel_size]
        # Build network
        net = nn.ModuleList([])
        for i in range(n_layers):
            net.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size[i],
                                 padding=kernel_size[i] // 2), stride=stride[i])
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