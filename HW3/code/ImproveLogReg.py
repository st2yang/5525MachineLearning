import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define paramaters for the model
learning_rate = 0.001
batch_size = 256
n_epochs = 25


# Define the convenience functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# Define the cnn
def cnn(X):
    X = tf.reshape(X, [-1, 28, 28, 1])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)

    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2 = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2, W_fc1) + b_fc1)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    logits = tf.matmul(h_fc1, W_fc2) + b_fc2
    return logits


# Step 1: Read in data
mnist = input_data.read_data_sets('datasets/mnist', one_hot=True)

# Step 2: create placeholders for features and labels
X = tf.placeholder(tf.float32, [batch_size, 784], name='image')
Y = tf.placeholder(tf.float32, [batch_size, 10], name='label')

# Step 3: create weights and bias
w = tf.Variable(tf.zeros([784, 10]), dtype=tf.float32, name='weights')
b = tf.Variable(tf.zeros([batch_size, 10]), dtype=tf.float32, name='biases')

# Step 4: build model
# the model that returns the logits.
logits = cnn(X)

# Step 5: define loss function
entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)
loss = tf.reduce_mean(entropy)

# Step 6: define training op
# using gradient descent to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    n_batches = int(mnist.train.num_examples / batch_size)
    for i in range(n_epochs):
        total_loss = 0

        for _ in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
            total_loss += loss_batch
        print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))

    print('Total time: {0} seconds'.format(time.time() - start_time))

    print('Optimization Finished!')

    # test the model
    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))  # need numpy.count_nonzero(boolarr) :(

    n_batches = int(mnist.test.num_examples / batch_size)
    total_correct_preds = 0

    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        accuracy_batch = sess.run(accuracy, feed_dict={X: X_batch, Y: Y_batch})
        total_correct_preds += accuracy_batch

    print('Accuracy {0}'.format(total_correct_preds / mnist.test.num_examples))
