""" Using convolutional net on MNIST dataset of handwritten digit
(http://yann.lecun.com/exdb/mnist/)
"""
from __future__ import print_function
from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

N_CLASSES = 10

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
mnist = input_data.read_data_sets("datasets/mnist", one_hot=True)

# Step 2: Define paramaters for the model
LEARNING_RATE = 0.001
BATCH_SIZE = 128
SKIP_STEP = 10
DROPOUT = 0.75
N_EPOCHS = 5

# Step 3: create placeholders for features and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# We'll be doing dropout for hidden layer so we'll need a placeholder
# for the dropout probability too
# Use None for shape so we can change the batch_size once we've built the graph
with tf.name_scope('Data'):
    X = tf.placeholder(tf.float32, [None, 784], name="X_placeholder")
    Y = tf.placeholder(tf.float32, [None, 10], name="Y_placeholder")

dropout = tf.placeholder(tf.float32, name='dropout')
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

# Step 4 + 5: create weights + do inference
# the model is conv -> relu -> pool -> conv -> relu -> pool -> fully connected -> softmax

utils.make_dir('checkpoints')
utils.make_dir('checkpoints/CNN')

with tf.variable_scope('conv1') as scope:
    # first, reshape the image to [BATCH_SIZE, 28, 28, 1] to make it work with tf.nn.conv2d
    # use the dynamic dimension -1
    images = tf.reshape(X, shape=[-1, 28, 28, 1])

    # TO DO

    # create kernel variable of dimension [5, 5, 1, 32]
    # use tf.truncated_normal_initializer()
    kernel = tf.get_variable('kernel', [5, 5, 1, 32],
                             initializer=tf.truncated_normal_initializer())

    # TO DO

    # create biases variable of dimension [32]
    # use tf.constant_initializer(0.0)
    biases = tf.get_variable('biases', [32],
                             initializer=tf.random_normal_initializer())

    # TO DO

    # apply tf.nn.conv2d. strides [1, 1, 1, 1], padding is 'SAME'
    conv = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME')

    # TO DO

    # apply relu on the sum of convolution output and biases
    conv1 = tf.nn.relu(conv + biases, name=scope.name)

    # output is of dimension BATCH_SIZE x 28 x 28 x 32

with tf.variable_scope('pool1') as scope:
    # apply max pool with ksize [1, 2, 2, 1], and strides [1, 2, 2, 1], padding 'SAME'
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')
    # TO DO

    # output is of dimension BATCH_SIZE x 14 x 14 x 32

with tf.variable_scope('conv2') as scope:
    # similar to conv1, except kernel now is of the size 5 x 5 x 32 x 64
    kernel = tf.get_variable('kernels', [5, 5, 32, 64],
                             initializer=tf.truncated_normal_initializer())
    biases = tf.get_variable('biases', [64],
                             initializer=tf.random_normal_initializer())
    conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv + biases, name=scope.name)

    # output is of dimension BATCH_SIZE x 14 x 14 x 64

with tf.variable_scope('pool2') as scope:
    # similar to pool1
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')

    # output is of dimension BATCH_SIZE x 7 x 7 x 64

with tf.variable_scope('fc') as scope:
    # use weight of dimension 7 * 7 * 64 x 1024
    input_features = 7 * 7 * 64

    # create weights and biases

    # TO DO
    w = tf.get_variable('weights', [input_features, 1024],
                        initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases', [1024],
                        initializer=tf.constant_initializer(0.0))

    # reshape pool2 to 2 dimensional
    pool2 = tf.reshape(pool2, [-1, input_features])

    # apply relu on matmul of pool2 and w + b
    fc = tf.nn.relu(tf.matmul(pool2, w) + b, name='relu')

    # TO DO

    # apply dropout
    fc = tf.nn.dropout(fc, dropout, name='relu_dropout')

with tf.variable_scope('softmax_linear') as scope:
    # this you should know. get logits without softmax
    # you need to create weights and biases

    # TO DO
    w = tf.get_variable('weights', [1024, N_CLASSES],
                        initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases', [N_CLASSES],
                        initializer=tf.random_normal_initializer())
    logits = tf.matmul(fc, w) + b

# Step 6: define loss function
# use softmax cross entropy with logits as the loss function
# compute mean cross entropy, softmax is applied internally
with tf.name_scope('Loss'):
    # you should know how to do this too

    # TO DO
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits)
    loss = tf.reduce_mean(entropy, name='loss')

# Step 7: define training op
# using gradient descent with learning rate of LEARNING_RATE to minimize cost
# don't forgot to pass in global_step

# TO DO
with tf.name_scope('Optimizer'):
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)

# evaluation the model
with tf.name_scope("Evaluation"):
    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

# Create a summary to monitor loss
tf.summary.scalar("accuracy", accuracy)
tf.summary.scalar("loss", loss)

# merge summaries per collection
training_summary = tf.summary.merge_all()

with tf.Session() as sess:
    writer_train = tf.summary.FileWriter('./graphs/CNN', sess.graph)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # You have to create folders to store checkpoints
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/CNN/'))
    # if that checkpoint exists, restore from checkpoint
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    initial_step = global_step.eval()

    start_time = time.time()
    n_batches = int(mnist.train.num_examples / BATCH_SIZE)

    total_loss = 0.0
    for index in range(initial_step, n_batches * N_EPOCHS):  # train the model n_epochs times
        X_batch, Y_batch = mnist.train.next_batch(BATCH_SIZE)
        _, loss_batch, train_summ = sess.run([optimizer, loss, training_summary],
                                             feed_dict={X: X_batch, Y: Y_batch, dropout: DROPOUT})
        writer_train.add_summary(train_summ, index)
        total_loss += loss_batch
        if (index + 1) % SKIP_STEP == 0:
            print('Average loss at step {}: {}'.format(index + 1, total_loss / SKIP_STEP))
            total_loss = 0.0
            saver.save(sess, 'checkpoints/CNN/', index + 1)

    print("Optimization Finished!")  # should be around 0.35 after 25 epochs
    print("Total time: {0} seconds".format(time.time() - start_time))

    # Test model calculate accuracy
    print('Accuracy on test data:', accuracy.eval(
        {X: mnist.test.images, Y: mnist.test.labels, dropout: 1.0}))
    print('Run the command line: tensorboard --logdir=./graphs/CNN')
