import numpy as np
import tensorflow as tf


class ConvLayer:

    '''
    This class will implement a convolutional layer.
    As input it will take the kernel size, stride and padding for the
    convolution, the number of output channels and the activation function.
    '''


    def __init__(self, kernel_size,
            stride, padding, channels, activation_function = None
        ):

        # assign the given parameters
        self.k_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.channels_out = channels
        self.activation_function = activation_function



    def __call__(self, input):

        '''
        This function will receive the input to the ConvLayer
        and compute the output.
        The input is a 4-D Tensor with shape (batch_size, height, width,
        channels).
        The output is also a 4-D Tensor.
        '''

        # get the batch size from the input
        self.batch_size = int(input.get_shape()[0])
        self.channels_in = int(input.get_shape()[3])


        # creating the weight and the bias tensor
        self.weights = tf.Variable(
                         tf.truncated_normal(
                           shape = [self.k_size,
                                self.k_size,
                                self.channels_in,
                                self.channels_out
                                ],
                           stddev = 0.1
                         )
                       )

        self.biases = tf.Variable(tf.constant(1.0,
                                        shape=[self.channels_out]
                                  )
                      )


        # convolution
        conv = tf.nn.conv2d(
                    input,
                    self.weights,
                    strides=[1, self.stride, self.stride, 1],
                    padding=self.padding)

        conv = conv + self.biases


        if self.activation_function == 'ReLU':
            return tf.nn.relu(conv)
        return conv
