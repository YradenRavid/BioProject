import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

path = "/home/u12784/bio/BioProject"

#define a simple convolutional layer
def conv_layer(input, channels_in, channels_out, name = "conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5,5,channels_in,channels_out], stddev = 0.1), name = "W")
        b = tf.Variable(tf.constant(0.1 , shape = [channels_out]), name = "b")
        conv = tf.nn.conv2d(input, w, strides=[1,1,1,1],padding="SAME")
        act = tf.nn.relu(conv +b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return  tf.nn.max_pool(act,ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding= "SAME")

#fully connected layer
def fc_layer(input, channels_in, channels_out, name = "FC"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([channels_in,channels_out], stddev = 0.1), name = "W")
        b = tf.Variable(tf.constant(0.1,shape = [channels_out]), name = "b")
        act = tf.nn.relu(tf.matmul(input,w)+b)
        return act



tf.reset_default_graph()
sess = tf.Session()

#Set placeholders and reshape the data:
x = tf.placeholder(tf.float32, shape= [None, 784], name = "input")
y = tf.placeholder(tf.float32, shape = [None, 10], name = "labels")
x_image = tf.reshape(x, [-1,28,28,1])
tf.summary.image("input",x_image,3)

# Create the network

#Convulotional layer
conv1 = conv_layer(x_image, 1, 32, "conv1")
conv2 = conv_layer(conv1, 32, 64, "conv2")
flattened = tf.reshape(conv2, [-1, 7 * 7 * 64])

# fully connected layer
fc1 = fc_layer(flattened, 7 * 7 * 64, 1024 ,"fc1")
logits = fc_layer(fc1, 1024, 10, "fc2")

#cross entropy - as loss function
with tf.name_scope("cross_entrupy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y))
    tf.summary.scalar("cross_entropy", cross_entropy)

#Use Adam optimizer to train the network
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#compute the accuracy
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.arg_max(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar("accuracy", accuracy)

#combine all summaries:
merged_summary = tf.summary.merge_all()

#initialize all the variables
sess.run(tf.global_variables_initializer())

#tensorboard
writer = tf.summary.FileWriter(path + '/graph')
writer.add_graph(sess.graph)

#Train for 2001 step
for i in range(2001):
    batch = mnist.train.next_batch(100)
    if i % 5 == 0:
        [train_accuracy,s] = sess.run([accuracy,merged_summary], feed_dict = {x: batch[0], y: batch[1]})
        writer.add_summary(s, i)

    #Occasionally report accuracy
    if i % 500 == 0:
        print("step %d, training accuracy %g" % (i, train_accuracy))

    # Run the training step
    sess.run(train_step, feed_dict = {x: batch[0], y: batch[1]})
