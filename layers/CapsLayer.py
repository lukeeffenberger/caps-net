import numpy as np
import tensorflow as tf


class CapsLayer:

    '''
    This class will implement a capsule layer.
    As input it will take the number and dims of the capsules for
    both layers.
    Additionally the number of routing iterations that should
    be used has to be set.
    In the call function the input tensor will be given.
    '''

    def __init__(self, count1, dim1, count2, dim2, rout_iter):

        # assigning the given parameters for the layers
        self.count1 = count1
        self.dim1 = dim1
        self.count2 = count2
        self.dim2 = dim2
        self.rout_iter = rout_iter



    def __call__(self, input):

        '''
        This function will receive the input to the CapsLayer
        and compute the output.
        The input is a 3-D Tensor with shape (batch_size, count1, dim1).
        The output is a 3-D Tnesor with shape (batch_size, count2, dim2).
        '''

        # check if shapes from input agree with given count1 and dim1
        if(input.get_shape()[1] != self.count1 or
           input.get_shape()[2] != self.dim1):
            raise ValueError('Input should have shape (?,{},{}) but has shape {}'.format(self.count1,
                                                                                        self.dim1,
                                                                                        input.get_shape()
                                                                                        )
                            )

        # get the batch size from the input
        self.batch_size = int(input.get_shape()[0])

        c1, d1 = self.count1, self.dim1
        c2, d2 = self.count2, self.dim2

        # creating the weight tensor
        self.weights = tf.Variable(
                         tf.truncated_normal(
                           shape = [c1, c2, d1, d2],
                           stddev = 0.1
                         )
                       )


        # compute the prediction vectors
        prediction_vectors = self.predict_vectors(input)

        return self.routing(prediction_vectors)



    # squashing function from paper
    def squash(self, tensor):

        # tensor with same dimensions as the tensor with the length of the
        # vector along the specified axis stored in every component of this
        # vector, norm is the euclidean norm here

        norm = tf.norm(tensor, keep_dims=True, axis=2)
        normed_tensor = tensor/norm

        squashing_factor = norm**2/(1+norm**2)

        return squashing_factor * normed_tensor



    def predict_vectors(self, inp):

        '''
        Gets the weights and the input into the right dimension and
        returns the matrix multiplication.
        '''

        bs = self.batch_size
        weights = self.weights
        c1, d1 = self.count1, self.dim1
        c2, d2 = self.count2, self.dim2



        # reshape the weights and input
        inp = tf.reshape(inp, shape = [bs, c1, 1, d1, 1])
        inp = tf.tile(inp, multiples = [1, 1, c2, 1, 1])
        weights = tf.expand_dims(weights, axis=0)
        weights = tf.tile(weights, multiples = [bs, 1, 1, 1, 1])

        prediction_vectors = tf.matmul(weights, inp, transpose_a=True)

        return tf.squeeze(prediction_vectors)




    def routing(self, prediction_vectors):

        '''
        Does the routing algorithm and outputs the capsules
        of the next layer.
        '''

        c1, d1 = self.count1, self.dim1
        c2, d2 = self.count2, self.dim2

        logits = tf.zeros(shape = [self.batch_size, c1, c2])

        for i in range(self.rout_iter):

            # compute the coupling coefficients
            coupling_coeffs = tf.nn.softmax(logits)

            # reshape coupling coefficients
            coupling_coeffs = tf.expand_dims(coupling_coeffs, axis=-1)
            coupling_coeffs = tf.tile(coupling_coeffs, [1, 1, 1, d2])

            # compute the input
            drive = tf.reduce_sum(
                      coupling_coeffs*prediction_vectors,
                      axis=1
                    )

            activation = self.squash(drive)


            # if it is not the last iteration comput the
            # agreement and update the logits
            if (i != self.rout_iter-1):

                # reshape activation
                activation = tf.expand_dims(activation, axis=1)
                activation = tf.tile(activation, multiples=[1, c1, 1, 1])

                # compute agreement
                agreement = tf.reduce_sum(prediction_vectors*
                                          activation, axis=-1)

                # update the logits
                logits = logits + agreement

            else:
                return activation
