import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import deepchem as dc 
from sklearn.metrics import accuracy_score

np.random.seed(456)
tf.set_random_seed(456)

_, (train, valid, test), _  = dc.molnet.load_tox21()
train_X, train_y, train_w = train.X, train.y, train.w
valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
test_X, test_y, test_w = test.X, test.y, test.w


# Remove extra tasks
train_y = train_y[:, 0]
valid_y = valid_y[:, 0]
test_y = test_y[:, 0]
train_w = train_w[:, 0]
valid_w = valid_w[:, 0]
test_w = test_w[:, 0]

# network parameters
d = 1024
n_hidden = 50
learning_rate = 0.001
n_epochs = 10
batch_size = 100
dropout_prob = 1.0



with tf.name_scope("placeholder"):
    x = tf.placeholder(tf.float32,(None,d))
    y = tf.placeholder(tf.float32,(None,))
    keep_prob = tf.placeholder(tf.float32)

with tf.name_scope("hidden_laye"):
    W = tf.Variable(tf.random_normal((d,n_hidden)))
    b = tf.Variable(tf.random_normal((1,)))
    x_hidden = tf.nn.relu(tf.matmul(x,W) + b)
    x_hidden = tf.nn.dropout(x_hidden, keep_prob)

