from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from layers import conv2d
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from time import localtime, time

import tensorflow as tf
import os
import argparse
import sys


def reshape_data(data):
    images = data.images
    labels = data.labels
    reshaped_image = images.reshape(images.shape[0], 1, 28, 28)
    return reshaped_image, labels


def build_graph(x, dropout_rate):

    with tf.name_scope('transpose'):
        x_trans = tf.transpose(x, perm=[0, 2, 3, 1])

    is_training = tf.placeholder(tf.bool, name='is_training')

    with tf.name_scope('convolution_layers'):
        conv1 = conv2d(x_trans, filters=16, training=is_training, name='convolution1')
        conv2 = conv2d(conv1, filters=32, training=is_training, name='convolution2')
        max_pooling1 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2, name='max_pooling1')

        conv3 = conv2d(max_pooling1, filters=64, training=is_training, name='convolution3')
        conv4 = conv2d(conv3, filters=128, training=is_training, name='convolution4')
        max_pooling2 = tf.layers.max_pooling2d(conv4, pool_size=[2, 2], strides=2, name='max_pooling2')

    with tf.name_scope('fully_connected'):
        flatten = tf.layers.flatten(max_pooling2)
        fc = tf.layers.dense(flatten, units=100, activation=tf.nn.relu, name='fc',
                             kernel_initializer=tf.random_uniform_initializer(minval=-0.2, maxval=0.2))

        fc_drop = tf.layers.dropout(fc, rate=dropout_rate, training=is_training, name='dropout')

        logits = tf.layers.dense(fc_drop, units=10, name='logits',
                                 kernel_initializer=tf.random_uniform_initializer(minval=-0.2, maxval=0.2))

    return logits, is_training


def main(_):

    # get current time
    t = tuple(localtime(time()))[1:5]

    # load FASHION-MNIST dataset
    data = input_data.read_data_sets('data/fashion', one_hot=True)

    # get training, validation, test data and reshape data
    train_images, train_labels = reshape_data(data.train)
    validation_images, validation_labels = reshape_data(data.validation)
    test_images, test_labels = reshape_data(data.test)

    # training data generator with data augmentation
    train_datagen = ImageDataGenerator(rotation_range=30,
                                       horizontal_flip=True,
                                       vertical_flip=True)
    train_datagen.fit(train_images)

    # Shapes of training dataset
    print("Training set images shape: {shape}".format(shape=data.train.images.shape))
    print("Training set labels shape: {shape}".format(shape=data.train.labels.shape))

    # Shapes of validation dataset
    print("Validation set images shape: {shape}".format(shape=data.validation.images.shape))
    print("Validation set labels shape: {shape}".format(shape=data.validation.labels.shape))

    # Shapes of test dataset
    print("Test set images shape: {shape}".format(shape=data.test.images.shape))
    print("Test set labels shape: {shape}".format(shape=data.test.labels.shape))

    # summary directory path to save summaries
    summary_dir = '/tmp/FASHION-MNIST/run-%02d%02d-%02d%02d' % t

    # checkpoint directory path
    ckpt_dir = 'checkpoint'

    x = tf.placeholder(tf.float32, [None, 1, 28, 28])
    y = tf.placeholder(tf.float32, [None, 10])

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.lr, global_step, 300, 0.95, staircase=True)

    with tf.name_scope('logits'):
        logits, is_training = build_graph(x, FLAGS.dropout_rate)
        prediction = tf.nn.softmax(logits)

    with tf.name_scope('loss'):
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        tf.summary.scalar('cross_entropy', loss_op)

    with tf.name_scope('optimizer'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss_op, global_step)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    # model saver
    saver = tf.train.Saver()

    # Start training
    with tf.Session() as sess:

        # training and validation dataset summary writer
        train_writer = tf.summary.FileWriter(summary_dir + '/train', sess.graph)
        validation_writer = tf.summary.FileWriter(summary_dir + '/validation', sess.graph)

        # Run the initializer
        sess.run(init)

        max_validation_acc = 0.9
        for step, (batch_x, batch_y) in enumerate(train_datagen.flow(train_images,
                                                                     train_labels,
                                                                     batch_size=FLAGS.batch_size)):

            _, summary = sess.run([train_op, merged], feed_dict={x: batch_x,
                                                                 y: batch_y,
                                                                 is_training: True})
            train_writer.add_summary(summary, step)

            if step % 50 == 0:
                # calculate batch loss and accuracy
                loss, acc, summary = sess.run([loss_op, accuracy, merged], feed_dict={x: validation_images,
                                                                                      y: validation_labels,
                                                                                      is_training: False})
                validation_writer.add_summary(summary, step)

                print("Step {0}, Minibatch Loss= {1}, Validation Accuracy= {2}".format(str(step), "{:.4f}".format(loss),
                                                                                       "{:.3f}".format(acc)))

                if acc > max_validation_acc:
                    # save model
                    saver.save(sess, os.path.join(ckpt_dir, 'ckpt'), global_step=step)
                    max_validation_acc = acc

            if step >= FLAGS.num_steps:
                break

        print("Optimization Finished!")

        # restore best model
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = ckpt.model_checkpoint_path
            saver.restore(sess, ckpt_name)
            print('Restore {}'.format(ckpt_name))

        # calculate test accuracy
        test_accuracy = sess.run(accuracy, feed_dict={x: test_images,
                                                      y: test_labels,
                                                      is_training: False})
        print("Test Accuracy: {}".format(test_accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int,
                        default=32,
                        help='the batch size')
    parser.add_argument('--lr', type=float,
                        default=0.001,
                        help='the initial learning rate')
    parser.add_argument('--dropout_rate', type=float,
                        default=0.3,
                        help='the dropout rate between 0 and 1')
    parser.add_argument('--num_steps', type=int,
                        default=2000,
                        help='the number of training steps')
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
