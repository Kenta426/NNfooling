""" Hyper Parameter
"""

class Params(object):
    # key for visualization
    key2char = {"14":"E", "15":"F", "17":"H", "18":"I", "29":"T"}
    char2key = {"E":"14", "F":"15", "H":"17", "I":"18", "T":"29"}

    # location of the data
    train_data = 'data/train_data.pkl'
    train_labels = 'data/train_labels.pkl'

    # input dimension
    image_dim = (28, 28)
    channel = 1

    # model configuration
    convolution_layer = 1
    num_filter = 2

    # learning
    learning_rate = 0.01
