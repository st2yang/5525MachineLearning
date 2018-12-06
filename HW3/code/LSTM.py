import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Training Parameters
learning_rate = 0.001
training_steps = 3000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 28  # MNIST data input (img shape: 28*28)
timesteps = 28  # timesteps
num_hidden = 128  # hidden layer num of features
num_classes = 10  # MNIST total classes (0-9 digits)


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, _ = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights) + biases


mnist = input_data.read_data_sets('datasets/mnist', one_hot=True)

# tf Graph input
with tf.name_scope("Input"):
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])

# Define weights
with tf.name_scope("Trainable"):
    w = tf.Variable(tf.random_normal([num_hidden, num_classes]))
    b = tf.Variable(tf.random_normal([num_classes]))

logits = RNN(X, w, b)

with tf.name_scope("Loss"):
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))

with tf.name_scope("Optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)

# evaluate the model
with tf.name_scope("Evaluation"):
    prediction = tf.nn.softmax(logits)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Create a summary to monitor loss
tf.summary.scalar("accuracy", accuracy)
tf.summary.scalar("loss", loss_op)

# merge summaries per collection
training_summary = tf.summary.merge_all()

# Start training
with tf.Session() as sess:
    writer_train = tf.summary.FileWriter('./graphs/LSTM', sess.graph)

    sess.run(tf.global_variables_initializer())
    start_time = time.time()

    for step in range(1, training_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # Run optimization op (backprop)
        _, train_summ = sess.run([optimizer, training_summary], feed_dict={X: batch_x, Y: batch_y})
        writer_train.add_summary(train_summ, step)
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", minibatch loss= " + "{:.4f}".format(loss) +
                  ", training accuracy= {:.3f}".format(acc))

    print("Optimization Finished!")
    print("Total time: {0} seconds".format(time.time() - start_time))

    # Test model calculate accuracy
    test_data = mnist.test.images.reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
    print('Run the command line: tensorboard --logdir=./graphs/LSTM')
