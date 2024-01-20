from predictor import UnetPredictor

import os
from PIL import Image

HOME = os.getcwd()
DATA = os.path.join(HOME, 'data')
CHECKPOINTS = os.path.join(HOME, 'checkpoints')
CHECKPOINT = os.path.join(CHECKPOINTS, 'best.pth')
TEST_DATA = os.path.join(DATA, 'test-images')


def main():
    predictor = UnetPredictor(checkpoint=CHECKPOINT)

    for filename in os.listdir(TEST_DATA):
        print(filename)
        full_filename = os.path.join(TEST_DATA, filename)
        image = Image.open(full_filename)

        mask = predictor.predict(image, out_threshold=0.5)
        predictor.plot_mask(image, mask)


if __name__ == "__main__":
    main()
