"""
usuful functions
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
import random

# for the assignment, we will only use horizontal and vertical aligned letters
# E, F, H, I, T
key2char = {"14":"E", "15":"F", "17":"H", "18":"I", "29":"T"}
char2key = {"E":"14", "F":"15", "H":"17", "I":"18", "T":"29"}

def show_letter(img, n = 5, resize = False):
    """ display n images
    """
    fig, axs = plt.subplots(1,n, figsize=(15, 6))
    axs = axs.ravel()
    for i in range(n):
        if resize:
            im = imresize(img[i].reshape(28,28).T, (10,10))
        else:
            im = img[i].reshape(28,28).T
        axs[i].imshow(im, cmap=plt.cm.Greys)
        axs[i].axis('off')
    plt.show()


def show_smaple(data, label, n = 5):
    """ display random n images from label
    """
    images, labels = data
    img = images[np.where(labels == char2key[label])]
    idx = random.sample(range(img.shape[0]), n)
    img = img[idx]
    fig, axs = plt.subplots(1,n, figsize=(15, 6))
    axs = axs.ravel()
    for i in range(n):
#         im = imresize(img[i].reshape(28,28).T, (10,10))
        im = img[i].reshape(28,28).T
        axs[i].imshow(im, cmap=plt.cm.Greys)
        axs[i].axis('off')
    plt.show()

def one_hot(key, arr):
    order = {}
    for i, k in enumerate(key):
        order[k] = i
    values = [order[a] for a in arr]
    out = np.eye(len(key))
    return out[values]

def fool_target(key, arr):
    true = np.argmax(arr)
    print (true)
    r = random.randint(0, len(arr)-1)
    while r==true:
        r = random.randint(0, len(arr)-1)
    out = np.eye(len(key))
    return out[true], out[r]
