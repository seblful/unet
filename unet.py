import torch
from torch import nn

from torchvision.transforms.functional import center_crop


class DoubleConvolution(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=0):
        super().__init__()

        self.conv_first = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    padding='same')

        self.conv_second = nn.Conv2d(in_channels=out_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     padding='same')

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_first(x)
        x = self.relu(x)
        x = self.conv_second(x)
        x = self.relu(x)

        return x


class MaxPool(nn.Module):
    def __init__(self,
                 kernel_size=2,
                 stride=2):
        super().__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size,
                                     stride=stride)

    def forward(self, x):
        x = self.max_pool(x)

        return x


class TranspConvolution(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 stride=2):
        super().__init__()

        self.transp_conv = nn.ConvTranspose2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride)

    def forward(self, x):
        x = self.transp_conv(x)
        return x


class CroppAndConcat(nn.Module):
    def forward(self, x, contracting_x):
        cropped_x = center_crop(contracting_x, [x.shape[2], x.shape[3]])
        x = torch.cat((x, cropped_x), dim=1)

        return x


class UnetEncoder(nn.Module):
    def __init__(self,
                 in_channels):
        super().__init__()

        self.double_conv_mod = nn.ModuleList([DoubleConvolution(i, o) for i, o in [(
            in_channels, 64), (64, 128), (128, 256), (256, 512)]])

        self.num_conv = len(self.double_conv_mod)

        self.max_pool = MaxPool()

    def forward(self, x):
        skip_connections = []

        for i in range(self.num_conv):
            # Pass x to double convolution
            x = self.double_conv_mod[i](x)

            # Append x to skip connections
            skip_connections.append(x)

            # Pass x to max pool
            x = self.max_pool(x)

        return x, skip_connections


class UnetDecoder(nn.Module):
    def __init__(self,
                 out_channels):
        super().__init__()

        self.downsampling_dims = [
            (1024, 512), (512, 256), (256, 128), (128, 64)]

        self.num_conv = len(self.downsampling_dims)

        self.transp_conv_mod = nn.ModuleList(
            [TranspConvolution(i, o) for i, o in self.downsampling_dims])

        self.crop_cat = CroppAndConcat()

        self.double_conv_mod = nn.ModuleList(
            [DoubleConvolution(i, o) for i, o in self.downsampling_dims])

        self.last_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, skip_connections):
        for i in range(self.num_conv):
            # Pass x to transpose convolution
            x = self.transp_conv_mod[i](x)

            # Concatenate x with cropped skip connection
            x = self.crop_cat(x, skip_connections.pop())

            # Pass x to double convolution
            x = self.double_conv_mod[i](x)

        # Pass x to last conv
        x = self.last_conv(x)

        return x


class UNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()

        self.encoder = UnetEncoder(in_channels=in_channels)
        self.middle_conv = DoubleConvolution(512, 1024)
        self.decoder = UnetDecoder(out_channels=out_channels)

    def forward(self, x):
        # Pass x to encoder
        x, skip_connections = self.encoder(x)

        # Pass x to middle conv
        x = self.middle_conv(x)

        # Pass x to decoder
        x = self.decoder(x, skip_connections)

        return x
