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
        with open(head_dir + 'train_data.pkl', 'wb') as f:
            pickle.dump(images, f)
        with open(head_dir + 'train_labels.pkl', 'wb') as f:
            pickle.dump(labels, f)
    return images, labels

if __name__ == '__main__':
    FILE = '../data/emnist-balanced-train.csv'
    print("loading file...")
    start = time.time()
    im, l = load_data(FILE)
    print("{} seconds to load".format(time.time()-start))
    print("saving")
    filter_data((im, l))
    # print(list(char2key.keys()))
