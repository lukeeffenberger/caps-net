import numpy as np
import tensorflow as tf

class ConvCapsLayer:

    '''
    This class will implement a convolution to capsule layer.
    As input it will take the kernel size, stride and padding for the
    convolution and the dimensions and number of channels of the capusles.
    '''

    def __init__(self, kernel_size, stride, padding, dimension, channels):

        # assign the given parameters
        self.k_size = kernel_size
        self.stride = stride
        self.dim = dimension
        self.channels_out = channels
        self.padding = padding

    def __call__(self, input):

        '''
        This function will receive the input to the ConvCapsLayer
        and compute the output.
        The input is a 4-D Tensor with shape (batch_size, height, width,
        channels).
        The output is a 3-D Tensor with shape (batch_size,
        height2*width2*channels_o, dim).
        '''

        # get the batch size from the input
        self.batch_size = tf.shape(input)[0]
        self.channels_in = int(input.get_shape()[3])

        # creating the weight and the bias tensor
        self.weights = tf.Variable(
                         tf.truncated_normal(
                           shape = [self.k_size,
                                self.k_size,
                                self.channels_in,
                                self.dim * self.channels_out
                                ],
                           stddev = 0.1
                         )
                       )

        self.biases = tf.Variable(tf.constant(1.0,
                                        shape=[self.dim*self.channels_out]
                                  )
                      )

        # convolution
        conv = tf.nn.conv2d(
                    input,
                    self.weights,
                    strides=[1, self.stride, self.stride, 1],
                    padding=self.padding)
        conv = conv + self.biases

        # read out the height and the with of the output
        self.height = int(conv.get_shape()[1])
        self.width = int(conv.get_shape()[2])

        # reshape to capsules and squash
        capsules = tf.reshape(conv, shape=[self.batch_size,
                                    self.height*self.width*self.channels_out,
                                    self.dim
                                    ]
                    )
        capsules = self.squash(capsules)

        return capsules


    # squashing function from paper
    def squash(self, tensor):

        # tensor with same dimensions as the tensor with the length of the
        # vector along the specified axis stored in every component of this
        # vector, norm is the euclidean norm here


        norm = tf.norm(tensor, keepdims=True, axis=2)
        normed_tensor = tensor/norm

        squashing_factor = norm**2/(1+norm**2)

        return squashing_factor * normed_tensor
