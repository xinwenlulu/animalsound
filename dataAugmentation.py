import tensorflow as tf
import numpy as np
import common.augmentation.specaugment as specaugment


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, x_train, y_train, batch_size=64, timemask = 24, shuffle=True):
        self.batch_size = batch_size
        self.x = x_train
        self.y = y_train
        self.indices = range(y_train.shape[0])
        self.shuffle = shuffle
        self.timemask = timemask
        self.on_epoch_end()

    def __len__(self):
        return self.y.shape[0] // self.batch_size

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]

        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(self.y.shape[0])
        if self.shuffle:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        X = np.empty((self.batch_size,  500, 128, 1), dtype=np.float32)
        y = np.empty((self.batch_size, 30), dtype=np.float32)

        for i, idx in enumerate(batch):
            original = tf.transpose(self.x[idx, ])
            sa = specaugment.SpecAugment(original.numpy(), timemask=self.timemask)
            _ = sa.time_mask()
            specaugX = sa.freq_mask()
            X[i, ] = tf.transpose(specaugX)
            y[i] = self.y[idx, ]

        return X, y




