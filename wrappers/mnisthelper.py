import os
import struct
import numpy as np

class MNIST():
    """Wrapper to load the MNIST data.

    This wrapper helps us to load the MNIST data and to chunk it into batches.
    In parts it is from Lukas' '02_mnist-helper.py'.
    """

    def __init__(self, directory):
        """Load all binaries into numpy arrays."""

        # Save the directory, where the MNIST data is stored.
        self._directory = directory
        # Load the binaries into numpy arrays.
        self._training_images = self._load_binaries("./train-images.idx3-ubyte")
        self._training_labels = self._load_binaries("./train-labels.idx1-ubyte")
        self._test_images = self._load_binaries("./t10k-images.idx3-ubyte")
        self._test_labels = self._load_binaries("./t10k-labels.idx1-ubyte")
        # Read out the number of training samples.
        self._training_samples_n = self._training_labels.shape[0]

    def _load_binaries(self, file_name):
        """Transform a specific binary file into a numpy array."""
        # Get the complete path to the binary file.
        path = os.path.join(self._directory, file_name)
        # Open the file.
        with open(path, 'rb') as fd:
            # Read out the number of items and the codes.
            check, items_n = struct.unpack(">ii", fd.read(8))
            # If it is an image.
            if "images" in file_name and check == 2051:
                # Read out height and width.
                height, width = struct.unpack(">II", fd.read(8))
                # Read out the images.
                images = np.fromfile(fd, dtype = 'uint8')
                # Reshape the images to (items_n, height, width).
                images = np.reshape(images, (items_n, height, width))
                return images
            # If it is an label.
            elif "labels" in file_name and check == 2049:
                # Read out the labels
                labels = np.fromfile(fd, dtype = 'uint8')
                return labels
            # If it is not an image and not an label.
            else:
                # Throw error.
                raise ValueError("Not a MNIST file: " + path)

    def get_training_batch(self, batch_size):
        """Generator to provide training batches."""
        # Create random indices for shuffling the data.
        random_indices = np.random.choice(
                                    a=self._training_samples_n,
                                    size=self._training_samples_n,
                                    replace = False
                         )
        # Shuffle the images and the labels.
        training_images = self._training_images[random_indices]
        training_labels = self._training_labels[random_indices]
        # For the number of batches.
        for i in range(self._training_samples_n // batch_size):
            # Compute the start and the end point of the batch.
            on = i * batch_size
            off = on + batch_size
            # Create batch.
            batch = training_images[on:off], training_labels[on:off]
            yield batch

    def get_validation_batch(self, batch_size):
        """Get the validation batch.

        As it is to validate the performance it should always be the same batch.
        """
        batch = self._test_images[:batch_size], self._test_labels[:batch_size]
        return batch
