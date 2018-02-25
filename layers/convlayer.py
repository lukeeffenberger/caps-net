import numpy as np
import tensorflow as tf


class ConvLayer:
    """This class will implement a simple convolutional layer.

    As input it will take the parameters for the convolution (kernel size,
    stride and padding), the number of channels it should have and the
    acitvation function that should be applied.
    """

    def __init__(
            self, kernel_size,stride, padding, channels, activation_function = None
        ):
        """Assign the parameters."""
        self.k_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.channels_out = channels
        self.activation_function = activation_function

    def __call__(self, input):
        """Computes the output of the layer.

        The input is a 4-D Tensor with shape [batch_size, height_in, width_in,
        channels_in).
        The output is a 4-D Tensor with shape [batch_size, height_out, width_out,
        channels_out).
        """
        # Read out the batch size and the number of channels coming in from
        # the input.
        self.batch_size = tf.shape(input)[0]
        self.channels_in = int(input.get_shape()[3])

        # Create the weights and the bias tensor for the convolution.
        self.weights = tf.Variable(
                         tf.truncated_normal(
                           shape = [
                                self.k_size,
                                self.k_size,
                                self.channels_in,
                                self.channels_out
                           ],
                           stddev = 0.01
                         )
                       )
        self.biases = tf.Variable(tf.constant(1.0, shape=[self.channels_out]))

        # Convolution.
        conv = tf.nn.conv2d(
                    input=input,
                    filter=self.weights,
                    strides=[1, self.stride, self.stride, 1],
                    padding=self.padding)
        conv = conv + self.biases
        # Apply the given activation function.
        if self.activation_function == 'ReLU':
            return tf.nn.relu(conv)
        else:
            return conv
