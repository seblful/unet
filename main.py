from unet import UnetDecoder, DoubleConvolution, UpSample
import numpy as np
import torch


def main():
    a = UpSample(in_channels=1024, out_channels=512)
    data = torch.randn(size=(1, 1024, 28, 28))
    a.forward(data)


if __name__ == "__main__":
    main()
