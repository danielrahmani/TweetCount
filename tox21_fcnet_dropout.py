import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

np.random.seed(456)
tf.set_random_seed(456)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# _, (train, valid, test), _  = dc.molnet.load_tox21()
# train_X, train_y, train_w = train.X, train.y, train.w
# valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
# test_X, test_y, test_w = test.X, test.y, test.w


# # Remove extra tasks
# train_y = train_y[:, 0]
# valid_y = valid_y[:, 0]
# test_y = test_y[:, 0]
# train_w = train_w[:, 0]
# valid_w = valid_w[:, 0]
# test_w = test_w[:, 0]

# network parameters
d = np.shape(mnist.train.images)
n_hidden = 50
learning_rate = 0.001
n_epochs = 10
batch_size = 100
dropout_prob = 1.0

label_size = np.shape(mnist.train.labels)[1]

with tf.name_scope("placeholder"):
    x = tf.placeholder(tf.float32,(None,d))
    y = tf.placeholder(tf.float32,(None,label_size))
    keep_prob = tf.placeholder(tf.float32)

with tf.name_scope("hidden_layer"):
    W = tf.Variable(tf.random_normal((d,n_hidden)))
    b = tf.Variable(tf.random_normal((n_hidden,)))
    x_hidden = tf.nn.relu(tf.matmul(x,W) + b)
    x_hidden = tf.nn.dropout(x_hidden, keep_prob)
    
with tf.name_scope("output"):
    W = tf.Variable(tf.random_normal((n_hidden,label_size)))
    b = tf.Variable(tf.random_normal((label_size,)))
    y_logit = tf.matmul(x_hidden,W) + b
    y_pred = tf.nn.softmax(y_logit)
#     y_one_prob = tf.sigmoid(y_logit)
#     y_pred = tf.round(y_one_probe)
    
with tf.name_scope("loss"):
    y_expand = tf.expand_dims(y,1)
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit,label=y_expand)
    l = tf.reduce_sum(entropy)
    
with tf.name_scope("optim"):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(l)
    
with tf.name_scope("summeries"):
    tf.summary.scalar("loss",l)
    merged = tf.summary.merge_all()
    
train_writer = tf.summary.FileWriter("fcnet-tox21-dropout")


N = train_X.shape[0]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        batch_X, batch_y = mnist.train.next_batch(batch_size=batch_size)
        feed_dict = {x:batch_X,y:batch_y,keep_prob:dropout_prob}
        _,summary,loss = sess.run([train_op,merged,l],feed_dict=feed_dict)
        print(f"epoch {epoch}, loss: {loss}")
        train_writer.add_summary(summary,loss)
        step += 1
        pos += batch_size
    
    train_y_pred = sess.run(y_pred,feed_ditc={x:train_X,keep_prob:1.0})
    valid_y_ppred = sess.run(y_pred,feed_dict={x_valid,keep_prob:1.0})
    test_y_pred = sess.run(y_pred.feed_dict={x:x_test,keep_prob:1.0})
    
train_weighted_score = accuracy_score(train_y, train_y_pred, sample_weight=train_w)
print("Train Weighted Classification Accuracy: {train_weighted_score}")
valid_weighted_score = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
print("Valid Weighted Classification Accuracy: {valid_weighted_score}")
test_weighted_score = accuracy_score(test_y, test_y_pred, sample_weight=test_w)
print("Test Weighted Classification Accuracy: {test_weighted_score}")
