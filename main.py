# https://note.com/pc_python/n/necf0d54fbab6

from os import makedirs
from os.path import dirname, join

import cv2
import numpy as np


def main():
    def mosaic(img, alpha):
        h, w, ch = img.shape

        img = cv2.resize(img, (int(w * alpha), int(h * alpha)))
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

        return img

    def decreaseColor(img):
        dst = img.copy()

        idx = np.where((0 <= img) & (32 > img))
        dst[idx] = 16
        idx = np.where((32 <= img) & (64 > img))
        dst[idx] = 48
        idx = np.where((64 <= img) & (96 > img))
        dst[idx] = 80
        idx = np.where((96 <= img) & (128 > img))
        dst[idx] = 112
        idx = np.where((128 <= img) & (160 > img))
        dst[idx] = 144
        idx = np.where((160 <= img) & (192 > img))
        dst[idx] = 176
        idx = np.where((192 <= img) & (224 > img))
        dst[idx] = 208
        idx = np.where((224 <= img) & (256 >= img))
        dst[idx] = 240

        return dst

    base_dir = dirname(__file__)
    dist_dir = join(base_dir, "dist")
    try:
        makedirs(dist_dir)
    except FileExistsError:
        pass

    img = cv2.imread(join(base_dir, "capy.png"))

    dst = mosaic(img, 0.2)
    dst = decreaseColor(dst)

    cv2.imwrite(join(dist_dir, "capy_processed.png"), dst)


main()
