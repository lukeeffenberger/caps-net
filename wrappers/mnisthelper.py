import os
import struct
import numpy as np

class MNIST():
    def __init__(self, directory):
        self._directory = directory
        self._training_images = self._load_binaries("./train-images.idx3-ubyte")
        self._training_labels = self._load_binaries("./train-labels.idx1-ubyte")
        self._test_images = self._load_binaries("./t10k-images.idx3-ubyte")
        self._test_labels = self._load_binaries("./t10k-labels.idx1-ubyte")
        self._training_samples_n = self._training_labels.shape[0]

    def _load_binaries(self, file_name):
        path = os.path.join(self._directory, file_name)

        with open(path, 'rb') as fd:
            check, items_n = struct.unpack(">ii", fd.read(8))

            if "images" in file_name and check == 2051:
                height, width = struct.unpack(">II", fd.read(8))
                images = np.fromfile(fd, dtype = 'uint8')
                return np.reshape(images, (items_n, height, width))
            elif "labels" in file_name and check == 2049:
                return np.fromfile(fd, dtype = 'uint8')
            else:
                raise ValueError("Not a MNIST file: " + path)

    def get_training_batch(self, batch_size):
        random_indices = np.random.choice(
                                    a=self._training_samples_n,
                                    size=self._training_samples_n,
                                    replace = False
                         )
        training_images = self._training_images[random_indices]
        training_labels = self._training_labels[random_indices]
        for i in range(self._training_samples_n // batch_size):
            on = i * batch_size
            off = on + batch_size
            yield training_images[on:off], training_labels[on:off]

    def get_validation_batch(self, batch_size):
        return self._test_images[:batch_size], self._test_labels[:batch_size]
