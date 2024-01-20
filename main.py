from dataset import CancerDataset
from trainer import UnetTrainer

import os
HOME = os.getcwd()
DATA = os.path.join(HOME, 'data')
CHECKPOINTS = os.path.join(HOME, 'checkpoints')


def main():
    cancer_dataset = CancerDataset(data_path=DATA,
                                   target_imgsz=512)

    unet_trainer = UnetTrainer(dataset=cancer_dataset,
                               val_ratio=0.2,
                               batch_size=4,
                               num_epochs=20,
                               checkpoints_path=CHECKPOINTS,
                               lr=0.0001)

    unet_trainer.train()


if __name__ == "__main__":
    main()
