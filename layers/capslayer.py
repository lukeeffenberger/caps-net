import numpy as np
import tensorflow as tf


class CapsLayer:
    """This class will implement a capsule layer.

    As input it will take the number and dimension of the capsules in the "higher"
    layer. Additionally you have to specify the number of routing iterations.
    """

    def __init__(self, count2, dim2, rout_iter):
        """Assign the given parameters."""
        self.count2 = count2
        self.dim2 = dim2
        self.rout_iter = rout_iter

    def __call__(self, input):
        """Computes the output of the layer.

        The input is a 3-D Tensor with shape [batch_size, count1, dim1].
        The output is a 3-D Tensor with shape [batch_size, count2, dim2].
        """

        # Read out batch size, and number and dimension of "lower" layer
        # capsules from the input.
        self.batch_size = tf.shape(input)[0]
        self.count1 = int(input.get_shape()[1])
        self.dim1 = int(input.get_shape()[2])
        # Create the weights tensor.
        self.weights = tf.Variable(
                         tf.truncated_normal(
                           shape = [self.count1, self.count2, self.dim1, self.dim2],
                           stddev = 0.01
                         )
                       )
        # Compute the prediction vectors.
        prediction_vectors = self.predict_vectors(input)
        # Compute the output of the layer with the routing algotithm.
        output = self.routing(prediction_vectors)
        return output

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



    def predict_vectors(self, inp):
        """Compute prediction vectors.

        Computes the prediction vectors from each "lower" layer capsules
        for each "higher" layer capsules.
        """

        # Reshape the weights and the input to be able to do to nice matrix
        # multiplication for obtaining the prediction vectors.
        inp = tf.reshape(inp, shape = [self.batch_size, self.count1, 1, self.dim1, 1])
        inp = tf.tile(inp, multiples = [1, 1, self.count2, 1, 1])
        weights = tf.expand_dims(self.weights, axis=0)
        weights = tf.tile(weights, multiples = [self.batch_size, 1, 1, 1, 1])
        # Compute the prediction vectors.
        prediction_vectors = tf.matmul(weights, inp, transpose_a=True)
        # Delete the artifical dimensions.
        prediction_vectors = tf.squeeze(prediction_vectors)
        return prediction_vectors

    def routing(self, prediction_vectors):
        """Implements the routing algorithm, computes the output of the layer."""

        # Initialize the logits (b_ij in the paper).
        logits = tf.zeros(shape = [self.batch_size, self.count1, self.count2])

        # For the number of specified routing iterations.
        for i in range(self.rout_iter):
            # Compute the coupling coefficients (c_ij in the paper) from the logits.
            coupling_coeffs = tf.nn.softmax(logits)
            # Reshape coupling coefficients to the same shape as the prediction
            # vectors allowing component-wise multiplication.
            coupling_coeffs = tf.expand_dims(coupling_coeffs, axis=-1)
            coupling_coeffs = tf.tile(coupling_coeffs, [1, 1, 1, self.dim2])

            # Compute the input/drive (s_j in the paper) to the "higher"
            # layer caspule.
            drive = tf.reduce_sum(
                      coupling_coeffs*prediction_vectors,
                      axis=1
                    )
            # Compute the activation (v_j in the paper) of the "higher" layer
            # capsule.
            activation = self.squash(drive)

            # If it is not the last iteration do one routing step.
            if (i != self.rout_iter-1):
                # Reshape activation to the same shape as the prediction
                # vectors allowing component-wise multiplication.
                activation = tf.expand_dims(activation, axis=1)
                activation = tf.tile(activation, multiples=[1, self.count1, 1, 1])
                # Compute the agreement (a_ij in the paper) between the
                # prediction vectors and the current activation.
                agreement = tf.reduce_sum(prediction_vectors*
                                          activation, axis=-1)
                # Update the logits.
                logits = logits + agreement
            # If it is the last iteration return the activation of the capsule
            # as output.
            else:
                return activation
