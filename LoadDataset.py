import cv2
import numpy as np
import requests

from datasets import get_valid_transforms


def loader(params):
    image_nparray = np.asarray(bytearray(requests.get(params).content), dtype=np.uint8)
    image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
    # image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transforms = get_valid_transforms()
    augmented = transforms(image=image)
    image = augmented['image']
    return image