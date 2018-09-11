import numpy as np
from scipy.misc import imread, imsave


def crop(img, path):
    if img.shape[0] > img.shape[1]:
        mid = img.shape[0] // 2
        diff = 150
        img = img[mid - diff: mid + diff, :, :]
        prop = img.shape[0] / img.shape[1]
    else:
        mid = img.shape[1] // 2
        diff = 150
        img = img[:, mid - diff: mid + diff, :]
        prop = img.shape[1] / img.shape[0]

    if prop > 1.3:
        diff = (300 - img.shape[0]) // 2
        fill = np.zeros((diff, img.shape[1], img.shape[2]), dtype=np.uint8)
        fill.fill(255)
        img = np.vstack((fill, img, fill))

    imsave('./test/' + path.split('/')[6], img)


with open('/home/caique/datasets/caltech101/caltech101_test.txt') as f:
    paths = f.read().split('\n')

for path in paths:
    img = imread(path)
    if img.shape[0] > 300 or img.shape[1] > 300:
        crop(img, path)
