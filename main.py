from dataset import CancerDataset
from unet import UNet
from trainer import UnetTrainer
import torch

import os
HOME = os.getcwd()
DATA = os.path.join(HOME, 'data')
CHECKPOINTS = os.path.join(HOME, 'checkpoints')


def main():
    # unet = UNet(in_channels=3,
    #             out_channels=3)
    # data = torch.randn(size=(1, 3, 572, 572))

    # x = unet(data)

    d = CancerDataset(data_path=DATA)
    # print(d[0]['image'].shape, d[0]['mask'].shape)

    t = UnetTrainer(dataset=d,
                    val_ratio=0.1,
                    batch_size=2,
                    num_epochs=5,
                    checkpoints_path=CHECKPOINTS,
                    amp=True)

    t.train()


if __name__ == "__main__":
    main()
