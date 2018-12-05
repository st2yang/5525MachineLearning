""" Starter code for logistic regression model
with MNIST in TensorFlow
MNIST dataset: yann.lecun.com/exdb/mnist/
"""
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 25

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
mnist = input_data.read_data_sets('datasets/mnist', one_hot=True)

# Step 2: create placeholders for features and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# there are 10 classes for each image, corresponding to digits 0 - 9.
# Features are of the type float, and labels are of the type int
with tf.name_scope("Input"):
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])

# Step 3: create weights and bias
# weights and biases are initialized to 0
# shape of w depends on the dimension of X and Y so that Y = X * w + b
# shape of b depends on Y
with tf.name_scope("Trainable"):
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
# to get the probability distribution of possible label of the image
# DO NOT DO SOFTMAX HERE
with tf.name_scope("Model"):
    logits = tf.add(tf.matmul(X, w), b, name='logits')

# Step 5: define loss function
# use cross entropy loss of the real labels with the softmax of logits
# use the method:
# tf.nn.softmax_cross_entropy_with_logits(logits, Y)
# then use tf.reduce_mean to get the mean loss of the batch
with tf.name_scope("Loss"):
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=Y, name="entropy")
    loss = tf.reduce_mean(entropy, name='loss')

# Step 6: define training op
# using gradient descent to minimize loss
with tf.name_scope("Optimizer"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# evaluate the model
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
    writer_train = tf.summary.FileWriter('./graphs/logistic', sess.graph)

    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    n_batches = int(mnist.train.num_examples / batch_size)
    for epoch in range(n_epochs):  # train the model n_epochs times
        total_loss = 0

        for i in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch, train_summ = sess.run([optimizer, loss, training_summary],
                                                 feed_dict={X: X_batch, Y: Y_batch})
            writer_train.add_summary(train_summ, epoch * n_batches + i)
            total_loss += loss_batch
        print('Average loss epoch {0}: {1}'.format(epoch, total_loss / n_batches))

    print('Optimization Finished!')  # should be around 0.35 after 25 epochs
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # Test model calculate accuracy
    print('Accuracy on test data:', accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
    print('Run the command line: tensorboard --logdir=./graphs/logistic')
