import numpy as np
import tensorflow as tf


class DenseLayer:

    '''
    This class will implement a fully connected layer.
    As input it will take the number of neurons in the upper
    layer and the activation function.
    '''


    def __init__(self, n_out, activation_function = None):

        # assign the given parameters
        self.n_out = n_out
        self.activation_function = activation_function



    def __call__(self, input):

        '''
        This function will receive the input to the ConvLayer
        and compute the output.
        The input is a 2-D Tensor with shape (batch_size, n_in).
        The output is also a 4-D Tensor.
        '''

        # get the batch size from the input
        self.batch_size = int(input.get_shape()[0])
        self.n_in = int(input.get_shape()[1])


        # creating the weight and the bias tensor
        self.weights = tf.Variable(
                         tf.truncated_normal(
                           shape = [self.n_in, self.n_out],
                           stddev = 0.1
                         )
                       )

        self.biases = tf.Variable(tf.constant(1.0,
                                        shape=[self.n_out]
                                  )
                      )


        # calculate the drive
        drive = tf.matmul(input, self.weights) + self.biases


        if self.activation_function == 'ReLU':
            return tf.nn.relu(drive)
        elif self.activation_function == 'Sigmoid':
            return tf.nn.sigmoid(drive)
        return drive
