from dataset import CancerDataset
from unet import UNet
from trainer import UnetTrainer
import torch

import os
HOME = os.getcwd()
DATA = os.path.join(HOME, 'data')
CHECKPOINTS = os.path.join(HOME, 'checkpoints')


def main():
    cancer_dataset = CancerDataset(data_path=DATA)

    unet_trainer = UnetTrainer(dataset=cancer_dataset,
                               val_ratio=0.1,
                               batch_size=2,
                               num_epochs=5,
                               checkpoints_path=CHECKPOINTS,
                               amp=True)

    unet_trainer.train()


if __name__ == "__main__":
    main()
