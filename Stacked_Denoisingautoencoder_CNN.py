import tensorflow as tf
from autoencoder_models.Autoencoder_CNN import Autoencoder
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('./data',one_hot=True)
n_samples = int(mnist.train.num_examples)
training_epoch = 20
batch_size = 64
display_step = 1





ae1 = Autoencoder(filter_size=[5,5,1,32],
                  input_shape=[64,28,28,1])
ae2 = Autoencoder(filter_size=[3,3,32,64],
                  input_shape=[64,14,14,32])



x_ft = tf.placeholder(tf.float32,[None,28,28,1])
h = x_ft
h = ae1.hidden_transfer(
    tf.nn.conv2d(h,ae1.weights['w1'],[1,1,1,1],padding='SAME')
    +ae1.weights['b1']
)
h = tf.nn.max_pool(h,[1,2,2,1],[1,2,2,1],padding='SAME')
h = ae2.hidden_transfer(
    tf.nn.conv2d(h,ae2.weights['w1'],[1,1,1,1],padding='SAME')
    +ae2.weights['b1']
)
h = tf.nn.max_pool(h,[1,2,2,1],[1,2,2,1],padding='SAME')

h = ae2.hidden_transfer(
    tf.nn.conv2d_transpose(
        h,ae2.weights['w2'],
        ae2.hidden_conv.get_shape(),
        [1,2,2,1],padding='SAME'
    )
    +ae2.weights['b2']
)
h = tf.nn.conv2d(h,ae2.weights['w3'],[1,1,1,1],padding='SAME')+ae2.weights['b3']

h = ae1.hidden_transfer(
    tf.nn.conv2d_transpose(
        h,ae1.weights['w2'],
        ae1.hidden_conv.get_shape(),
        [1,2,2,1],padding='SAME'
    )
    +ae1.weights['b2']
)
h = tf.nn.conv2d(h,ae1.weights['w3'],[1,1,1,1],padding='SAME')+ae1.weights['b3']
cost = tf.reduce_mean(
    tf.square(h-x_ft)
)
train_step_ft = tf.train.AdamOptimizer(0.001).minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(training_epoch):
    avg_cost = 0
    total_batch = int(n_samples/batch_size)

    for i in range(total_batch):
        batch_xs,_=mnist.train.next_batch(batch_size)
        cost,_ = sess.run(ae1.partial_fit(),feed_dict={ae1.x:np.reshape(batch_xs,[-1,28,28,1])})
        avg_cost +=cost/n_samples*batch_size

    if epoch % display_step == 0:
        print("Epoch:{},Cost:{:.9f}".format(epoch,avg_cost))

print("************************First AE training finished****************************")

for epoch in range(training_epoch):
    avg_cost = 0
    total_batch = int(n_samples / batch_size)

    for i in range(total_batch):
        batch_xs, _ = mnist.train.next_batch(batch_size)
        h_ae1_out = sess.run(ae1.transform(),feed_dict={ae1.x:np.reshape(batch_xs,[-1,28,28,1])})
        cost = sess.run(ae2.partial_fit(),feed_dict={ae2.x:h_ae1_out})
        avg_cost += cost / n_samples * batch_size

    if epoch % display_step == 0:
        print("Epoch:{},Cost:{:.9f}".format(epoch, avg_cost))

print("************************Second AE training finished****************************")
print("Test accurancy before fine-tune")

print(sess.run(cost,feed_dict={x_ft:np.reshape(mnist.test.images[:64],[-1,28,28,1])}))





''''''