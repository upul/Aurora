import numpy as np
import gzip
import pickle
import os


class MNIST:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        train, valid, test = self._load_data()
        self.X_train, self.y_train = train[0], train[1]

        # encoding y_train using one-hot encoding
        self.y_train_one_hot = np.zeros((self.y_train.shape[0], 10))
        self.y_train_one_hot[np.arange(self.y_train.shape[0]), self.y_train] = 1

        self.X_valid, self.y_valid = valid[0], valid[1]
        self.X_test, self.y_test = test[0], test[1]

    def train_batch_generator(self):
        while True:
            rand_indices = np.random.choice(self.X_train.shape[0], self.batch_size, False)
            yield self.X_train[rand_indices], self.y_train_one_hot[rand_indices]

    def validation(self):
        return self.X_valid, self.y_valid

    def testing(self):
        return self.X_test, self.y_test

    def num_features(self):
        return self.X_train.shape[1]

    def _load_data(self):
        script_dir = os.path.dirname(__file__)
        mnist_file = os.path.join(os.path.join(script_dir, 'data'), 'mnist.pkl.gz')

        with gzip.open(mnist_file, 'rb') as mnist_file:
            u = pickle._Unpickler(mnist_file)
            u.encoding = 'latin1'
            train, val, test = u.load()
        return train, val, test
