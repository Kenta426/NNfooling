""" Preprocess the data
"""
import csv
import numpy as np
import time
import pickle

key2char = {"14":"E", "15":"F", "17":"H", "18":"I", "29":"T"}
char2key = {"E":"14", "F":"15", "H":"17", "I":"18", "T":"29"}

def load_data(file_dir):
    labels = []
    images = []
    with open(file_dir, 'r') as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            labels.append(np.array(row[0]))
            images.append(np.array(row[1:], dtype='float'))
    labels = np.array(labels)
    images = np.array(images)
    return images, labels

def filter_data(data, save = True):
    images, labels = data
    images = images[np.in1d(labels, list(key2char.keys()))]
    labels = labels [np.in1d(labels, list(key2char.keys()))]
    head_dir = '../data/'
    if save:
        with open(head_dir + 'test_data.pkl', 'wb') as f:
            pickle.dump(images, f)
        with open(head_dir + 'test_labels.pkl', 'wb') as f:
            pickle.dump(labels, f)
    return images, labels

def random_data(data, save = True):
    images, labels = data
    images = images[~np.in1d(labels, list(key2char.keys()))]
    labels = labels [~np.in1d(labels, list(key2char.keys()))]
    head_dir = '../data/'
    if save:
        with open(head_dir + 'random_data.pkl', 'wb') as f:
            pickle.dump(images, f)
        with open(head_dir + 'random_labels.pkl', 'wb') as f:
            pickle.dump(labels, f)
    return images, labels

def svd_reconstruction(data, num_s = 2, save = True):
    images, labels = data
    images = images[np.in1d(labels, list(key2char.keys()))]
    labels = labels [np.in1d(labels, list(key2char.keys()))]

    for i, im in enumerate(images):
        im = im.reshape(28,28).T
        U,s,V = np.linalg.svd(im)
        U = U[:, :num_s]
        V = V[:num_s]
        s = s[:num_s]
        s = np.diag(s)
        reconst = np.dot(np.dot(U, s), V)
        images[i] = reconst.T.flatten()
    # print (reconst.shape)

    head_dir = '../data/'
    
    # scale the image
    images = (images-np.min(images))/(np.max(images) - np.min(images))
    images = 255*images

    print (np.max(images), np.min(images))
    if save:
        with open(head_dir + 'test_data_s'+str(num_s)+'.pkl', 'wb') as f:
            pickle.dump(images, f)
    return images, labels


if __name__ == '__main__':
    FILE = '../data/emnist-balanced-test.csv'
    print("loading file...")
    start = time.time()
    im, l = load_data(FILE)
    print("{} seconds to load".format(time.time()-start))
    print("saving")
    data = filter_data((im, l), save = False)
    data = svd_reconstruction(data, num_s = 6, save = True)
    # print(list(char2key.keys()))
    # print (np.max(data[0]))
