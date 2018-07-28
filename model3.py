import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.metrics import average_precision_score

# local variables
TF_NUM = 13
ROWS_NUM=200000
EPOCHS = 5
BATCH_SIZE = 200
BUFFER = 200
#path = os.getcwd()
TB_path = "/home/u12784/bio/model2/BioProject"


def oneHot(string):
    trantab = str.maketrans('ACGT','0123')
    string = string+'ACGT'
    data = list(string.translate(trantab))
    return tf.keras.utils.to_categorical(data)[0:-4]   # .transpose()


def read_selex_csv_to_array(experiment_name, selex_num, rows_to_read, train_data=True):
    #path = os.path.join(os.path.expanduser('~'), 'hello-world','train/' if train_data else 'test/')
    path = "/home/u12784/bio/BioProject/TF13"
    #path = os.getcwd()
    while(1):
        selex_name = path +'/'+ experiment_name + '_selex_' + str(selex_num) + '.txt'
        if os.path.isfile(selex_name):
            break
        else:
            selex_num = selex_num-1
    print(selex_num)
    print("I stating to count rows")
    #rows_to_read = sum(1 for line in open(selex_name))
    print("I finished to count rows")
    selex = pd.read_csv(selex_name, sep="\t", header=-1, names=['seq', 'int'], nrows=rows_to_read)
    data = map(oneHot, selex['seq'])
    return np.array(list(data))

def read_pbm_to_array(ex_name, train_data=True):
    # path = os.path.join(os.path.expanduser('~'), 'hello-world', 'train/' if train_data else 'test/')
    path = "/home/u12784/bio/BioProject/TF13"
    # path = os.getcwd()
    pbm_name = path +'/'+ ex_name + '_pbm.txt'
    pbm = pd.read_csv(pbm_name, header=-1, names=['seq', 'int'])
    data = map(oneHot, pbm['seq'])
    test = np.array(list(data))
    return test[:, :36, :]
#    return np.append(test[:, :36, :], np.zeros([len(test), 40 - 36, 4]), axis=1)

def get_input_and_labels(n, max_selex_idx, rows_number):
    ex_name = 'TF' + str(n)
    test = read_pbm_to_array(ex_name, train_data=True)
    selex_0 = read_selex_csv_to_array(ex_name, selex_num=0, rows_to_read=rows_number, train_data=True)
    selex_4 = read_selex_csv_to_array(ex_name, selex_num=max_selex_idx, rows_to_read=rows_number, train_data=True)

    # create input data, labels
    input_data = np.append(selex_0, selex_4, axis=0)
    #input_data = np.append(input_data, np.zeros([len(input_data), 40 - selex_0.shape[1], 4]), axis=1)
    # input_data = np.append(input_data, np.zeros([len(input_data), 4, 40-selex_0.shape[2]]), axis=2)
    #labels = np.concatenate((np.zeros(len(selex_0)), np.ones(len(selex_4))))
    labels = np.zeros([len(input_data), 10])
    for i in range(0, 9):
        labels[1 + i * 100:(i + 1) * 100, i] = 1
    labels[901:, 9] = 1
    perm = np.random.permutation(labels.shape[0])
    seq_len = input_data.shape[1]
    test_len = test.shape[1]
    test = np.append(test[:,0:seq_len,:], test[:,test_len - seq_len: test_len,:], axis=0)
    return input_data[perm, :, :], labels[perm], test



# define a simple convolutional layer
def conv_layer(input, channels_in, channels_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([12, 1, channels_in, channels_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="b")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding="SAME")


# fully connected layer
def fc_layer(input, channels_in, channels_out, name="FC"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([channels_in, channels_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="b")
        act = tf.nn.relu(tf.matmul(input, w) + b)
        return act


def logits_to_perc(labels):
    clasiffier = tf.argmax(labels,axis=1)+1
    new_label = 1/clasiffier
    return new_label

# reset session and tensorboard
tf.reset_default_graph()
sess = tf.Session()

# get data
[input_data, labels, test] = get_input_and_labels(TF_NUM,6,ROWS_NUM)
train_data = [input_data, labels]
true_order = [int(x) for x in np.append(np.ones(200), np.zeros(len(test) - 200), axis=0)]
true_order = np.reshape(true_order,[len(true_order),1])
true_order = np.repeat(true_order,10,axis=1)
test_data = [test, true_order]


# Set placeholders and reshape the data:
x = tf.placeholder(tf.float32, shape=[None, input_data.shape[1] , 4], name="input")
y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
x_seq = tf.reshape(x, [-1, input_data.shape[1] * 4, 1, 1])


batch_size = tf.placeholder(tf.int64)
#tf.summary.image("input", x_seq, 3)

# Set database:
dataset = tf.data.Dataset.from_tensor_slices((x_seq, y)).repeat().batch(batch_size)
dataset = dataset.shuffle(buffer_size=BUFFER)

# init data
iter = dataset.make_initializable_iterator()
seqs, seqs_val = iter.get_next()



# Create the network

# Convulotional layer
conv1 = conv_layer(seqs, 1, 32, "conv1")
conv2 = conv_layer(conv1, 32, 64, "conv2")
sizelist=conv2.get_shape().as_list()
flat_len_c2d = sizelist[1]*sizelist[2]*sizelist[3]
flattened = tf.reshape(conv2, [-1, flat_len_c2d])

# fully connected layer
fc1 = fc_layer(flattened, flat_len_c2d, 128, "fc1")
logits = fc_layer(fc1, 128, 10, "fc2")



# cross entropy - as loss function
with tf.name_scope("cross_entrupy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=seqs_val))
    tf.summary.scalar("cross_entropy", cross_entropy)

# Use Adam optimizer to train the network
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# compute the accuracy
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(seqs_val, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

with tf.name_scope("AUPR"):
    test_label = logits_to_perc(logits)
    aupr_true = tf.reduce_max(seqs_val,axis=1)
    AUPR = average_precision_score(aupr_true,test_label)
    tf.summary.scalar("AUPR", AUPR)


# combine all summaries:
merged_summary = tf.summary.merge_all()

# initialize all the variables
sess.run(tf.global_variables_initializer())

# tensorboard
writer = tf.summary.FileWriter(TB_path + '/graph')
writer.add_graph(sess.graph)

sess.run(iter.initializer, feed_dict={ x: train_data[0], y: train_data[1], batch_size: BATCH_SIZE})
print('Training...')

# Train for 2001 step

for i in range(2001):
    sess.run(train_step, feed_dict={x: train_data[0], y: train_data[1], batch_size: BATCH_SIZE})
    if i % 5 == 0:
        # check accuracy and loss while taining
        [train_accuracy, loss ,s] = sess.run([accuracy, cross_entropy, merged_summary], feed_dict={x: train_data[0], y: train_data[1] , batch_size: BATCH_SIZE})
        # check test (AUPR of PBM)
        sess.run(iter.initializer,
                 feed_dict={x: test_data[0], y: test_data[1], batch_size: test_data[0].shape[0]})
        [test_AUPR, s] = sess.run([AUPR, merged_summary],
                                             feed_dict={x: test_data[0], y: test_data[1], batch_size: test_data[0].shape[0]})
        writer.add_summary(s, i)
    # Occasionally report accuracy
    if i % 500 == 0:
        print("step %d, training accuracy %g loss is %g AUPR %g" % (i, train_accuracy, loss, test_AUPR))

    # Run the training step


