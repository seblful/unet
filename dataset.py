import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CancerDataset(Dataset):
    def __init__(self,
                 data_path,
                 target_imgsz=512):

        self.frames_path = os.path.join(data_path, 'frames')
        self.masks_path = os.path.join(data_path, 'masks')

        self.frames_listdir = [os.path.join(self.frames_path, img_path) for img_path in os.listdir(
            self.frames_path) if os.path.splitext(img_path)[-1] in [".jpg", ".png"]]
        self.masks_listdir = [os.path.join(self.masks_path, img_path) for img_path in os.listdir(
            self.masks_path) if os.path.splitext(img_path)[-1] in [".jpg", ".png"]]

        self.target_imgsz = target_imgsz

    def load_image(self, image_path):
        return Image.open(image_path)

    def transform_image(self, image, is_mask=False):

        data_transform = transforms.Compose([
            transforms.Resize(size=(self.target_imgsz, self.target_imgsz)),
            transforms.ToTensor()])  # also normalization

        image = data_transform(image)

        image = image.float().contiguous()

        if is_mask:
            image, _ = torch.max(image, dim=0)
            image = image.unsqueeze(0)

        return image

    def __len__(self):
        return len(self.frames_listdir)

    def __getitem__(self, index):

        frame_name = self.frames_listdir[index]
        mask_name = self.masks_listdir[index]

        frame_image = self.load_image(frame_name)
        mask_image = self.load_image(mask_name)

        return {'image': self.transform_image(frame_image),
                'mask': self.transform_image(mask_image, is_mask=True)}
