
from predictor import UnetPredictor

import os
from PIL import Image

HOME = os.getcwd()
DATA = os.path.join(HOME, 'data')
CHECKPOINTS = os.path.join(HOME, 'checkpoints')
CHECKPOINT = os.path.join(CHECKPOINTS, 'checkpoint_epoch2.pth')
TEST_DATA = os.path.join(DATA, 'test-images')


def main():
    predictor = UnetPredictor(checkpoint=CHECKPOINT)

    for filename in os.listdir(TEST_DATA):
        full_filename = os.path.join(TEST_DATA, filename)
        image = Image.open(full_filename)

        transf_image, mask = predictor.predict(image)
        predictor.plot_mask(image, mask)

        break


if __name__ == "__main__":
    main()
