from unet import UNet

import torch
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt

import numpy as np


class UnetPredictor():
    def __init__(self,
                 checkpoint):

        self.transformer = transforms.Compose([
            transforms.Resize(size=(512, 512)),
            transforms.ToTensor()])  # also normalization

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Init model
        self.net = UNet(in_channels=3,
                        out_channels=1).to(self.device)
        state_dict = torch.load(checkpoint, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()

    def transform_image(self, image):
        image = self.transformer(image)
        image = image.float().contiguous().unsqueeze(0)
        image = image.to(self.device, dtype=torch.float32)

        return image

    def predict(self,
                image,
                out_threshold=0.5):
        # Transform image
        transf_image = self.transform_image(image)

        with torch.no_grad():
            output = self.net(transf_image).cpu()
            mask = torch.sigmoid(output)
            mask = mask > out_threshold
            mask = mask.view(512, 512)

        return mask

    def plot_mask(self, image, mask):
        fig, ax = plt.subplots(1, 2)
        ax[0].set_title('Input Image')
        ax[0].imshow(image)
        ax[1].set_title('Mask')
        ax[1].imshow(mask, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()
