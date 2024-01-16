
from unet import UNet

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms

HOME = os.getcwd()
DATA = os.path.join(HOME, 'data')
CHECKPOINTS = os.path.join(HOME, 'checkpoints')
CHECKPOINT = os.path.join(CHECKPOINTS, 'checkpoint_epoch5.pth')
TEST_DATA = os.path.join(DATA, 'test-images')


def transform_image(image):
    data_transform = transforms.Compose([
        transforms.Resize(size=(640, 640)),
        transforms.ToTensor()])  # also normalization

    image = data_transform(image)

    image = image.float().contiguous()

    return image


def predict_img(net,
                full_img,
                device,
                out_threshold=0.5):
    net.eval()
    img = transform_image(full_img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(
            output, (full_img.size[1], full_img.size[0]), mode='bilinear')

        mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()


def main():
    net = UNet(in_channels=3,
               out_channels=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    state_dict = torch.load(CHECKPOINT, map_location=device)
    net.load_state_dict(state_dict)

    for filename in os.listdir(TEST_DATA):
        full_filename = os.path.join(TEST_DATA, filename)
        img = Image.open(full_filename)

        mask = predict_img(net=net,
                           full_img=img,
                           device=device)

        print(mask.shape)

        plot_img_and_mask(np.array(img), mask.transpose(1, 2, 0))

        break


if __name__ == "__main__":
    main()
