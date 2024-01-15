from unet import UNet
import torch


def main():
    unet = UNet(in_channels=3,
                out_channels=3)
    data = torch.randn(size=(1, 3, 572, 572))

    x = unet(data)


if __name__ == "__main__":
    main()
