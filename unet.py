import torch
from torch import nn

from torchvision.transforms.functional import center_crop


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=0):
        super().__init__()

        self.conv_first = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    padding=padding)

        self.conv_second = nn.Conv2d(in_channels=out_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     padding=padding)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_first(x)
        x = self.relu(x)
        x = self.conv_second(x)
        x = self.relu(x)

        return x


class MaxPooling(nn.Module):
    def __init__(self,
                 kernel_size=2,
                 stride=2):
        super.__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size,
                                     stride=stride)

    def forward(self, x):
        x = self.max_pool(x)

        return x


class UpSample(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 stride=2):
        super().__init__()

        self.alt_conv = nn.ConvTranspose2d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride)

    def forward(self, x):
        print(x.shape)
        x = self.alt_conv(x)

        print(x.shape)

        return x


class CroppAndConcat(nn.Module):
    def forward(self, x, contracting_x):
        contracting_x = center_crop()
        # TO DO


class UnetDecoder(nn.Module):
    pass


class UnetEncoder(nn.Module):
    pass


class UNetNet(nn.Module):
    pass
