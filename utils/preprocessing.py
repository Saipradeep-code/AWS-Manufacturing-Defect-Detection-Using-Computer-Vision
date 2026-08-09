from PIL import Image
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def preprocess(image):
    image = image.convert("RGB").resize((224, 224))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    chw = arr.transpose(2, 0, 1)
    return (chw - MEAN) / STD


def get_transforms():
    return preprocess
