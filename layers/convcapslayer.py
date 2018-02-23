import numpy as np
import tensorflow as tf

class ConvCapsLayer:
    """This class will implement a Convolution to Capsule layer.

    As input it will take the parameters for the convolution (kernel size,
    stride and padding) and dimension and number of channels of capsules
    in the "higher" layer.
    """

    def __init__(self, kernel_size, stride, padding, dimension, channels):
        """Assign the given parameters."""
        self.k_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dim = dimension
        self.channels_out = channels

    def __call__(self, input):
        """Computes the output of the layer.

        The input is a 4-D Tensor with shape [batch_size, height_in, width_in,
        channels_in].
        It first does a normal convolution. Then cuts the output into chunks of
        of the dimension of the capsules and flattens the tensor.
        The output is a 3-D Tensor with shape [batch_size,
        height_out*width_out*channels_out, dimension].
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
                                self.dim * self.channels_out
                            ],
                           stddev = 0.1
                         )
                       )
        self.biases = tf.Variable(tf.constant(1.0, shape=[self.dim*self.channels_out]))

        # Convolution.
        conv = tf.nn.conv2d(
                    input=input,
                    filter=self.weights,
                    strides=[1, self.stride, self.stride, 1],
                    padding=self.padding)
        conv = conv + self.biases

        # Read out the height and the width of the convolution output. Might be
        # different than the height and width of the input, if padding was 'VALID'
        self.height = int(conv.get_shape()[1])
        self.width = int(conv.get_shape()[2])

        # Reshape the tensor to [batch_size, number_of_capsules, dimension].
        capsules = tf.reshape(conv, shape=[
                                    self.batch_size,
                                    self.height*self.width*self.channels_out,
                                    self.dim
                                    ]
                    )
        # Squash (from paper) the capsules.
        capsules = self.squash(capsules)
        return capsules

    def squash(self, tensor):
        """Implements the squashing function from paper."""

        # Compute the norm of all vectors. Will have the same dimension as
        # the input tensor.
        norm = tf.norm(tensor, keep_dims=True, axis=2)
        # Divide every component by the norm of the vector it belongs to.
        normed_tensor = tensor/norm
        # Compute the squashing factor.
        squashing_factor = norm**2/(1+norm**2)
        # Compute the squashed tensor.
        squashed_tensor = squashing_factor * normed_tensor
        return squashed_tensor
