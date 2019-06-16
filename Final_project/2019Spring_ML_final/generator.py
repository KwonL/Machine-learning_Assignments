import numpy as np
import keras
import os
from PIL import Image

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, target='Data/train_sequence/', batch_size=10, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_size = 10000
        self.target = target
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # return int(np.floor(self.dataset_size / self.batch_size))
        return 1000

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        images = np.zeros([self.batch_size, 20, 64, 64, 3])
        for i, idx in enumerate(indexes) :
            for t in range(20):
                img_path = os.path.join(self.target, 'sequence%04d' % idx, 'frames%02d.png' % t)
                img = np.array(Image.open(img_path)) / 255.0  # normalize
                images[i, t] = img

        X = images[:-1]
        Y = images[1:]

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.dataset_size)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


class ValidGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, target='Data/val_sequence/', batch_size=10, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_size = 500
        self.target = target
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.dataset_size / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        images = np.zeros([self.batch_size, 20, 64, 64, 3])
        for i, idx in enumerate(indexes) :
            for t in range(20):
                img_path = os.path.join(self.target, 'sequence%03d' % idx, 'frames%02d.png' % t)
                img = np.array(Image.open(img_path)) / 255.0  # normalize
                images[i, t] = img

        X = images[:-1]
        Y = images[1:]

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.dataset_size)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
