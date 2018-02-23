import numpy as np
import tensorflow as tf


class DenseLayer:
    """This class will implement a simple fully connected layer.

    As input it will take the number of neurons on the next layer and the
    acitvation function that should be applied.
    """

    def __init__(self, n_out, activation_function = None):
        """Assign the parameters."""
        self.n_out = n_out
        self.activation_function = activation_function

    def __call__(self, input):
        """Computes the output of the layer.

        The input is a 2-D Tensor with shape [batch_size, n_in].
        The output is also a 2-D Tensor with [batch_size, n_out].
        """
        # Read out the batch size and the number of channels coming in from
        # the input.
        self.batch_size = tf.shape(input)[0]
        self.n_in = int(input.get_shape()[1])

        # Create the weights and the bias tensor.
        self.weights = tf.Variable(
                         tf.truncated_normal(
                           shape = [self.n_in, self.n_out],
                           stddev = 0.01
                         )
                       )
        self.biases = tf.Variable(tf.constant(1.0, shape=[self.n_out]))

        # Calculate the drive.
        drive = tf.matmul(input, self.weights) + self.biases
        # Apply the given activation function.
        if self.activation_function == 'ReLU':
            return tf.nn.relu(drive)
        elif self.activation_function == 'Sigmoid':
            return tf.nn.sigmoid(drive)
        else:
            return drive
