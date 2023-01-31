from enum import Enum
from PIL import Image
import numpy as np


def loop_iterable(iterable):
    while True:
        yield from iterable


class NetArchitecture(Enum):
    """
    ENUM object for dropout option
    """
    ONE_FC = 1
    TWO_FC = 2
    THREE_FC = 3


class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""

    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)
