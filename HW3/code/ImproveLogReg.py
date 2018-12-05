import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define parameters for the model
learning_rate = 0.001
batch_size = 256
n_epochs = 5
skip_step = 100


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
with tf.name_scope("Data"):
    X = tf.placeholder(tf.float32, [None, 784], name='image')
    Y = tf.placeholder(tf.float32, [None, 10], name='label')

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

utils.make_dir('checkpoints')
utils.make_dir('checkpoints/CNNv2')

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
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

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
    writer_train = tf.summary.FileWriter('./graphs/CNNv2', sess.graph)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # You have to create folders to store checkpoints
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/CNNv2/'))
    # if that checkpoint exists, restore from checkpoint
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    initial_step = global_step.eval()

    start_time = time.time()
    n_batches = int(mnist.train.num_examples / batch_size)

    total_loss = 0.0
    for index in range(initial_step, n_batches * n_epochs):  # train the model n_epochs times
        X_batch, Y_batch = mnist.train.next_batch(batch_size)
        _, loss_batch, train_summ = sess.run([optimizer, loss, training_summary],
                                             feed_dict={X: X_batch, Y: Y_batch})
        writer_train.add_summary(train_summ, index)
        total_loss += loss_batch
        if (index + 1) % skip_step == 0:
            print('Average loss at step {}: {}'.format(index + 1, total_loss / skip_step))
            total_loss = 0.0
            saver.save(sess, 'checkpoints/CNNv2/', index + 1)

    print("Optimization Finished!")
    print("Total time: {0} seconds".format(time.time() - start_time))

    # Test model calculate accuracy
    print('Accuracy on test data:', accuracy.eval(
        {X: mnist.test.images, Y: mnist.test.labels}))
    print('Run the command line: tensorboard --logdir=./graphs/CNNv2')