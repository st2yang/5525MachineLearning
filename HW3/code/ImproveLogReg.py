import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define parameters for the model
learning_rate = 0.001
batch_size = 256
n_epochs = 3


# Define the cnn
def cnn(X):
    X = tf.reshape(X, [-1, 28, 28, 1])
    out = X

    channels = [32, 64]
    for i, c in enumerate(channels):
        with tf.variable_scope('block_{}'.format(i + 1)):
            out = tf.layers.conv2d(out, c, 5, padding='same')
            out = tf.nn.relu(out)
            out = tf.layers.max_pooling2d(out, 2, 2)

    out = tf.reshape(out, [-1, 7 * 7 * 64])
    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, 1024)
        out = tf.nn.relu(out)
    with tf.variable_scope('fc_2'):
        logits = tf.layers.dense(out, 10)

    return logits


# Step 1: Read in data
mnist = input_data.read_data_sets('datasets/mnist', one_hot=True)

# Step 2: create placeholders for features and labels
X = tf.placeholder(tf.float32, [None, 784], name='image')
Y = tf.placeholder(tf.float32, [None, 10], name='label')

# Step 3: build model
# the model that returns the logits.
logits = cnn(X)

# Step 4: define loss function
with tf.name_scope("Loss"):
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)
    loss = tf.reduce_mean(entropy)

# Step 5: define training op
# using gradient descent to minimize loss
with tf.name_scope("Optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# test the model
with tf.name_scope("Output"):
    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

# Create a summary to monitor loss
tf.summary.scalar("accuracy", accuracy)
tf.summary.scalar("loss", loss)

# merge summaries per collection
training_summary = tf.summary.merge_all()

with tf.Session() as sess:
    writer_train = tf.summary.FileWriter('./graphs/CNNv2', sess.graph)

    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    n_batches = int(mnist.train.num_examples / batch_size)
    for epoch in range(n_epochs):
        total_loss = 0

        for i in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch, train_summ = sess.run([optimizer, loss, training_summary],
                                                 feed_dict={X: X_batch, Y: Y_batch})
            writer_train.add_summary(train_summ, epoch * n_batches + i)
            total_loss += loss_batch
        print('Average loss epoch {0}: {1}'.format(epoch, total_loss / n_batches))

    print('Optimization Finished!')
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # Test model calculate accuracy
    print('Accuracy on test data:', accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
    print('Run the command line: tensorboard --logdir=./graphs/CNNv2')
