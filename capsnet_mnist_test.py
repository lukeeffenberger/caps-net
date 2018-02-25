import tensorflow as tf
from layers.capslayer import CapsLayer
from layers.convcapslayer import ConvCapsLayer
from layers.convlayer import ConvLayer
from layers.denselayer import DenseLayer
from wrappers.mnisthelper import MNIST

# Importing the mnist data with a wrapper, that provides generator for training
# and validation batches.
mnist_data = MNIST('./mnist_data')

# Testing Parameters
TEST_BATCH_SIZE = 600

def main():
    """ Testing CapsNet.

    The data flow graph is designed and the test session is run.
    """

    # Define the placeholders.
    image_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
    label_placeholder = tf.placeholder(dtype=tf.int64, shape=[None])
    # Read out the batch size from the placeholders.
    batch_size = tf.shape(image_placeholder)[0]

    # Convolutional layer.
    with tf.variable_scope('ReLU_Conv1'):
        # Add fourth dimension (number of channels) to the images, because it is
        # needed for convolution.
        image_reshaped = tf.expand_dims(image_placeholder, axis=-1)
        convolution = ConvLayer(
                        kernel_size = 9,
                        stride = 1,
                        padding = 'VALID',
                        channels = 256,
                        activation_function = 'ReLU'
                      )(image_reshaped)

    # Primary Caps layer. Does another convolution and cuts it into capsules.
    with tf.variable_scope('Primary_Caps'):
        primary_caps = ConvCapsLayer(
                            kernel_size = 9,
                            stride = 2,
                            padding = 'VALID',
                            dimension = 8,
                            channels = 32,
                        )(convolution)

    # Digit Caps layer. Basically the readout layer, that decides if the network
    # recognized a certain digit or not.
    with tf.variable_scope('Digit_Caps'):
        digit_caps = CapsLayer(
                            count2 = 10,
                            dim2 = 16,
                            rout_iter = 3
                     )(primary_caps)

    # Calculate the loss and the accuracy of the read out.
    with tf.variable_scope('Loss'):
        loss, accuracy = calculate_loss_accuracy(digit_caps, label_placeholder)



    # Define the tensorflow "saver" to restore the learned weights.
    saver = tf.train.Saver()

    # Session to train the model.
    with tf.Session() as sess:
         # Restore the weights from model.ckpt.
        saver.restore(sess, "./tmp/model.ckpt")

        # Get the tets batch.
        test_generator = mnist_data.get_test_batch(TEST_BATCH_SIZE)
        # Initialize list to store the accuracy of each batch.
        accuracies = []
        for image_samples, label_samples in test_generator:
            # Get the accuracy.
            _accuracy = sess.run(
                            [accuracy],
                            feed_dict = {image_placeholder: image_samples,
                                         label_placeholder: label_samples}
                                    )
            accuracies.append(_accuracy)

        # Get the overall test erro.
        test_error = 1 - np.mean(accuracies)        
        print("Test error: {}".format(test_error))

def calculate_loss_accuracy(digit_caps, labels):
    """Calculate the loss and the accuracy.

    The loss implements the loss from the paper. For more information check our
    report.
    Accuracy is computed by taking the digit with "longest" capsule, meaning the
    digit where the model is the most sure, that it is in the image, as the
    prediction of the network.
    """
    # Compute the length (euclidean norm) of the digit capsules.
    length_digit_caps = tf.norm(digit_caps, axis = 2)
    labels_one_hot = tf.one_hot(labels, depth=10)
    # Compute the false negative part of the loss.
    plus_loss =  labels_one_hot * tf.nn.relu(0.9 - length_digit_caps)
    # Compute the fals positive part of the loss.
    minus_loss = 0.5 * (1 - labels_one_hot) * tf.nn.relu(length_digit_caps - 0.1)
    # Compute the loss from those two parts
    loss = tf.reduce_sum(plus_loss + minus_loss, axis=-1)
    loss = tf.reduce_mean(loss)
    # Compute accuracy by comparing indices of longest capusle to the labels.
    correct_prediction = tf.equal(tf.argmax(length_digit_caps, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return loss, accuracy


if __name__ == '__main__':
    main()
