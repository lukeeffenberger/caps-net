import tensorflow as tf
import numpy as np
import argparse
from layers.capslayer import CapsLayer
from layers.convcapslayer import ConvCapsLayer
from layers.convlayer import ConvLayer
from layers.denselayer import DenseLayer
from wrappers.mnisthelper import MNIST

# Importing the mnist data with a wrapper, that provides generator for training,
# validation and test batches.
mnist_data = MNIST('./mnist_data')

# Training Parameters
EPOCHS = 30
TRAINING_BATCH_SIZE = 128
# Testing Parameters
TEST_BATCH_SIZE = 10

def main():
    """Training or testing CapsNet.

    The data flow graph is designed and the session is run.
    If the mode is 'train_on' the weights are loaded from 'model_weights/'.
    For both training modes the best model is saved in 'tmp/model_weights' and
    the summaries from the training process are written to 'tmp/summaries'.
    If the mode is 'test' the weights are loaded from 'model_weights/'.
    """
    # Get the mode.
    mode = parse_input()

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

    # Reconstructor that reconstructs the image from the representation of the
    # digit capsules. Consist of three dense layers.
    with tf.variable_scope('Dense1'):
        # Mask out (set to zero) all but the correct digit capsule and flatten
        # the tensor for the dense layer.
        digit_caps_flat = mask_and_flatten_digit_caps(digit_caps, label_placeholder)
        dense_1 = DenseLayer(
                        n_out = 512,
                        activation_function = 'ReLU'
                  )(digit_caps_flat)

    with tf.variable_scope('Dense2'):
        dense_2 = DenseLayer(
                        n_out = 1024,
                        activation_function = 'ReLU'
                  )(dense_1)

    with tf.variable_scope('Dense3'):
        dense_3 = DenseLayer(
                        n_out = 28*28,
                        activation_function = 'Sigmoid'
                  )(dense_2)
        # Reshape the output of this layer to same shape as the original image,
        # to obtain the reconstruction.
        reconstructions = tf.reshape(dense_3, shape=[batch_size, 28, 28])

    # Calculate the loss of the reconstruction.
    with tf.variable_scope('Reconstruction_Loss'):
        reconstruction_loss = calculate_reconstruction_loss(reconstructions, image_placeholder)

    # Set AdamOptimizer with default values (as described in the paper) as
    # optimizer. It minimizes the sum of the loss and the reconstruction loss.
    with tf.variable_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer()
        total_loss = loss + reconstruction_loss
        training_step = optimizer.minimize(total_loss)

    # Training modes
    if mode in ['train', 'train_on']:
        # Define which nodes to save, to display later in tensorboard.
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('reconstruction_loss', reconstruction_loss)
        tf.summary.scalar('total_loss', total_loss)
        merged_summaries = tf.summary.merge_all()

        # Define the tensorflow "writer" for tensorboard and the tensorflow "saver"
        # to save the learned weights.
        train_writer = tf.summary.FileWriter("./tmp/summaries/train", tf.get_default_graph())
        validation_writer = tf.summary.FileWriter("./tmp/summaries/validation", tf.get_default_graph())
        saver = tf.train.Saver()

        # Session to train the model.
        with tf.Session() as sess:
            # The training mode without pretrained weights.
            if mode == 'train':
                # Count steps for displaying in tensorboard.
                step = 0
                # Initialize best loss to infinity to be able to compare the current
                # validation loss to the best validation loss so far and store the
                # weights only if the mode was better.
                best_validation_loss = float('inf')
                # Initialize all variables.
                sess.run(tf.global_variables_initializer())
            # The training mode with pretrained weights.
            else:
                # Count steps for displaying in tensorboard.
                step = 30*np.ceil(60000.0/128.0)
                # Restore the weights.
                saver.restore(sess, "./model_weights/model.ckpt")
                # Initialize best loss to infinity to be able to compare the current
                # validation loss to the best validation loss so far and store the
                # weights only if the mode was better.
                best_validation_loss = 0.015

            # Train the network for the specified number of epochs.
            for epoch in range(EPOCHS):
                # Print current epoch number for verifying that the network trains.
                print("Epoch {}...".format(epoch))

                # Get the validation batch.
                image_samples, label_samples = mnist_data.get_validation_batch()
                # Validate the current performance.
                _summaries, _loss = sess.run(
                                    [merged_summaries, total_loss],
                                    feed_dict = {image_placeholder: image_samples,
                                                 label_placeholder: label_samples}
                                        )
                # Save the summaries.
                validation_writer.add_summary(_summaries, step)
                # Print current loss for eyeballing training process.
                print("Loss: {}".format(_loss))
                # Save weights, if the model had a lower validation loss than
                # in the previous epochs.
                if _loss < best_validation_loss:
                    save_path = saver.save(sess, "./tmp/model_weights/model.ckpt")
                    # Update the best loss so far.
                    best_validation_loss = _loss

                # Initialize a generator for training batches of specified size.
                training_generator = mnist_data.get_training_batch(TRAINING_BATCH_SIZE)
                # For all batches train the network.
                for image_samples, label_samples in training_generator:
                    _, _summaries  = sess.run(
                                        [training_step,merged_summaries],
                                        feed_dict = {image_placeholder: image_samples,
                                                     label_placeholder: label_samples}
                                      )
                    # Save the summaries.
                    train_writer.add_summary(_summaries, step)
                    # Count step one up.
                    step += 1

    # Testing mode.
    else:
        # Define the tensorflow "saver" to restore the learned weights.
        saver = tf.train.Saver()
        # Session to train the model.
        with tf.Session() as sess:
             # Restore the weights from model.ckpt.
            saver.restore(sess, "./model_weights/model.ckpt")
            # Get the test batch.
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
            # Get the overall test error.
            test_error = 1 - np.mean(accuracies)
            print("Test error: {}".format(test_error))

def mask_and_flatten_digit_caps(digit_caps, labels):
    """Mask out and flat the digit caps.

    All but the 16 values of the capsule corresponding to the correct label
    are set to 0. Then the digit caps are reshaped from [batch_size, 10, 16] to
    [batch_size, 10*16].
    """
    # Create a tensor in the same shape as the digit caps with ones at the
    # entries that are corresponding to the entries for the correct label
    # and zero everywhere else.
    labels = tf.one_hot(labels, depth=10)
    labels = tf.expand_dims(labels, axis=-1)
    labels = tf.tile(labels, [1,1,16])
    # Mask out the digit caps.
    masked_digit_caps = digit_caps * labels
    # Read out the batch size from the labels.
    batch_size = tf.shape(labels)[0]
    # Flat the digit caps.
    masked_and_flat = tf.reshape(masked_digit_caps, shape=[batch_size,10*16])
    return masked_and_flat

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

def calculate_reconstruction_loss(reconstructions, images):
    """Calculate the loss of the reconstruction."""
    # As the reconstructor has a sigmoid read out layer scale pixel intensties
    # of the images down to [0,1].
    images = images/255.0
    # Compute the sum squared error between reconstruction and original image.
    squared_error = tf.squared_difference(reconstructions, images)
    sum_squared_error = tf.reduce_sum(squared_error, axis=-1)
    # Scale down the reconstruction loss to let it not dominate the loss.
    reconstruction_loss = 0.0005 * 784 * tf.reduce_mean(sum_squared_error)
    return reconstruction_loss

def parse_input():
    """Parses the input.

    One can specify if the model should be trained from scratch (train), trained on with
    already pre-trained weights (train_on) or tested (test).
    """
    # Read in the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--mode',
                    help='Set the mode: train, train_on, test')
    args = parser.parse_args()
    # Check if the given argument for the mode is valid.
    if args.mode not in ['train', 'train_on', 'test']:
        # Raise an error if not.
        raise ValueError('The given mode "{}" is not a valid mode. Use "train",\
"train_on" or "test" instead.'.format(args.mode))
    return args.mode

if __name__ == '__main__':
    main()
