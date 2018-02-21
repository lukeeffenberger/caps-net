# import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from layers.capslayer import CapsLayer
from layers.convcapslayer import ConvCapsLayer
from layers.convlayer import ConvLayer
from layers.denselayer import DenseLayer
from wrappers.mnisthelper import MNIST

mnist_data = MNIST('./mnist_data')

#TRAINING PARAMS
EPOCHS = 50
BATCH_SIZE = 128

def main():
    tf.reset_default_graph()

    # define the placeholders
    image_placeholder = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 28, 28])
    label_placeholder = tf.placeholder(dtype=tf.int64, shape=[BATCH_SIZE])

    with tf.variable_scope('ReLU_Conv1'):
        image_reshaped = tf.expand_dims(image_placeholder, axis=-1)
        conv_layer = ConvLayer(
                        kernel_size = 9,
                        stride = 1,
                        padding = 'VALID',
                        channels = 256,
                        activation_function = 'ReLU'
                     )
        conv = conv_layer(image_reshaped)

    with tf.variable_scope('Primary_Caps'):
        primary_caps_layer = ConvCapsLayer(
                                kernel_size = 9,
                                stride = 2,
                                padding = 'VALID',
                                dimension = 8,
                                channels = 32,
                            )
        primary_caps = primary_caps_layer(conv)

    with tf.variable_scope('Digit_Caps'):
        digit_caps_layer = CapsLayer(
                                count1 = 6*6*32,
                                dim1 = 8,
                                count2 = 10,
                                dim2 = 16,
                                rout_iter = 3
                            )
        digit_caps = digit_caps_layer(primary_caps)

    with tf.variable_scope('Loss'):
        loss, accuracy = calculate_loss_accuracy(digit_caps, label_placeholder)

    #RECONSTRUCTOR
    with tf.variable_scope('Dense1'):
        digit_caps_flat = mask_and_flatten_digit_caps(digit_caps, label_placeholder)
        dense_1_layer = DenseLayer(
                            n_out = 512,
                            activation_function = 'ReLU'
                        )
        dense_1 = dense_1_layer(digit_caps_flat)

    with tf.variable_scope('Dense2'):
        dense_2_layer = DenseLayer(
                            n_out = 1024,
                            activation_function = 'ReLU'
                        )
        dense_2 = dense_2_layer(dense_1)

    with tf.variable_scope('Dense3'):
        dense_3_layer = DenseLayer(
                            n_out = 28*28,
                            activation_function = 'Sigmoid'
                        )
        dense_3 = dense_3_layer(dense_2)
        reconstructions = tf.reshape(dense_3, shape=[BATCH_SIZE, 28, 28])

    with tf.variable_scope('Reconstruction_Loss'):
        reconstruction_loss = calculate_reconstruction_loss(dense_3, image_placeholder)

    with tf.variable_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer()
        total_loss = loss + reconstruction_loss
        training_step = optimizer.minimize(total_loss)

    #SUMMARIES
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('reconstruction_loss', reconstruction_loss)
    tf.summary.scalar('total_loss', total_loss)
    merged_summaries = tf.summary.merge_all()

    #WRITER AND SAVER
    train_writer = tf.summary.FileWriter("./summaries/train", tf.get_default_graph())
    validation_writer = tf.summary.FileWriter("./summaries/validation", tf.get_default_graph())
    saver = tf.train.Saver()

    with tf.Session() as sess:
        step = 0
        best_validation_loss = float('inf')
        sess.run(tf.global_variables_initializer())

        for epoch in range(EPOCHS):
            print("Epoch {}...".format(epoch))

            training_generator = mnist_data.get_training_batch(BATCH_SIZE)
            for x, y in training_generator:
                _loss, _accuracy, _, _summaries  = sess.run([loss,
                                                accuracy,
                                                training_step,
                                                merged_summaries
                                                ],
                                                feed_dict = {image_placeholder: x,
                                                            label_placeholder: y})
                train_writer.add_summary(_summaries, step)
                step += 1

            validation_generator = mnist_data.get_validation_batch(BATCH_SIZE)
            for x, y in validation_generator:
                _summaries, _loss = sess.run([merged_summaries, total_loss],
                                        feed_dict = {image_placeholder: x,
                                                     label_placeholder: y})
                validation_writer.add_summary(_summaries, step)
                step += 1

            print("Loss: {}".format(_loss))
            if _loss < best_validation_loss:
                save_path = saver.save(sess, "./tmp/model.ckpt")
                best_validation_loss = _loss

def mask_and_flatten_digit_caps(digit_caps, labels):
    labels = tf.one_hot(labels, depth=10)
    labels = tf.expand_dims(labels, axis=-1)
    labels = tf.tile(labels, [1,1,16])
    masked_digit_caps = digit_caps * labels
    masked_and_flat = tf.reshape(masked_digit_caps, shape=[BATCH_SIZE,10*16])
    return tf.reshape(masked_digit_caps, shape=[BATCH_SIZE,10*16])

def calculate_loss_accuracy(digit_caps, labels):
    length_digit_caps = tf.norm(digit_caps, axis = 2)
    labels_one_hot = tf.one_hot(labels, depth=10)
    plus_loss =  labels_one_hot * tf.nn.relu(0.9 - length_digit_caps)
    minus_loss = 0.5 * (1 - labels_one_hot) * tf.nn.relu(length_digit_caps - 0.1)
    loss = tf.reduce_sum(plus_loss + minus_loss, axis=-1)
    loss = tf.reduce_mean(loss)
    correct_prediction = tf.equal(tf.argmax(length_digit_caps, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return loss, accuracy

def calculate_reconstruction_loss(reconstructions, images):
    images = images/255.0
    images_flatten = tf.reshape(images, shape=[BATCH_SIZE, 784])
    squared_error = tf.squared_difference(reconstructions, images_flatten)
    sse = tf.reduce_sum(squared_error, axis=-1)
    return 0.0005 * tf.reduce_mean(sse)

if __name__ == '__main__':
    main()
