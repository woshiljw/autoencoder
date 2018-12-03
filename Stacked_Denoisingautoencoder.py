from autoencoder_models.Autoencoder import Autoencoder
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./data',one_hot=True)

n_samples = int(mnist.train.num_examples)
training_epoch = 20
batch_size = 128
display_step = 1


n_inputs = 784
n_hidden1 = 400
n_hidden2 = 100
n_output = 10

ae1 = Autoencoder(
    n_layers=[n_inputs,n_hidden1],
    transfer_function=tf.nn.relu,
    optimizer=tf.train.AdamOptimizer(0.001)
)

ae2 = Autoencoder(
    n_layers=[n_hidden1,n_hidden2],
    transfer_function=tf.nn.relu,
    optimizer=tf.train.AdamOptimizer(0.001)
)

x = tf.placeholder(tf.float32,[None,n_hidden2])
w = tf.Variable(tf.zeros([n_hidden2,n_output]))
b = tf.Variable(tf.zeros([n_output]))
y = tf.matmul(x,w)+b

y_ = tf.placeholder(tf.float32,[None,n_output])

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_)
)

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

x_ft = tf.placeholder(tf.float32,[None,n_inputs])
h = x_ft

for layer in range(len(ae1.n_layers)-1):
    print(ae1.weights['encode'][layer]['w'].shape)
    h = ae1.transfer(
        tf.matmul(h,ae1.weights['encode'][layer]['w'])+ae1.weights['encode'][layer]['b']
    )

for layer in range(len(ae2.n_layers)-1):
    h = ae2.transfer(
        tf.matmul(h, ae2.weights['encode'][layer]['w']) + ae2.weights['encode'][layer]['b']
    )

y_ft = tf.matmul(h,w)+b
cross_entropy_ft = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=y_ft,labels=y_)
)
train_step_ft = tf.train.AdamOptimizer(0.001).minimize(cross_entropy_ft)
correct_prediction = tf.equal(tf.argmax(y_ft,1),tf.argmax(y_,1))
accurancy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(training_epoch):
    avg_cost = 0
    total_batch = int(n_samples/batch_size)

    for i in range(total_batch):
        batch_xs,_=mnist.train.next_batch(batch_size)
        cost = ae1.partial_fit(batch_xs)
        avg_cost +=cost/n_samples*batch_size

    if epoch % display_step == 0:
        print("Epoch:{},Cost:{:.9f}".format(epoch,avg_cost))

print("************************First AE training finished****************************")

for epoch in range(training_epoch):
    avg_cost = 0
    total_batch = int(n_samples / batch_size)

    for i in range(total_batch):
        batch_xs, _ = mnist.train.next_batch(batch_size)
        h_ae1_out = ae1.transform(batch_xs)
        cost = ae2.partial_fit(h_ae1_out)
        avg_cost += cost / n_samples * batch_size

    if epoch % display_step == 0:
        print("Epoch:{},Cost:{:.9f}".format(epoch, avg_cost))

print("************************Second AE training finished****************************")
for epoch in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(batch_size)
    h_ae1_out = ae1.transform(batch_xs)
    h_ae2_out = ae2.transform(h_ae1_out)
    _,cost = sess.run((train_step,cross_entropy),feed_dict={x:h_ae2_out,y_:batch_ys})
    if epoch % 10 == 0:
        print("Epoch:{},Cost:{:.9f}".format(epoch, cost))

print("************************Softmax layer finished****************************")

print("Test accurancy before fine-tune")
print(sess.run(accurancy,feed_dict={x_ft:mnist.test.images,y_:mnist.test.labels}))

