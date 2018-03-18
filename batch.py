import numpy as np
class Batch(object):
    def __init__(self, data, labels, batch):
        self.data = data
        self.labels = labels
        self.batch = batch

    def next_batch(self):
        return self.data[:self.batch], self.labels[:self.batch]

    def shuffle(self):
        idx = np.random.permutation(len(self.data))
        self.data = self.data[idx]
        self.labels = self.labels[idx]


if __name__ == '__main__':
    b = Batch(np.array([1,2,3,4,5,6,7,8,9,10]), np.array([0]*10), 6)
    d, _ = b.next_batch()
    print (d)
    d, _ = b.next_batch()
    print (d)
    d, _ = b.next_batch()
    print (d)
