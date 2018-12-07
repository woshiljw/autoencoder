import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import batch_norm
'''
class Stacked_AutoEncoder(object):
    def __init__(self):
        pass
'''
class Data(object):
    def __init__(self, datafile, batchsize):
        self.img = np.load(datafile)['train']
        self.train = self.img[:4480]
        self.train_index = np.arange(len(self.train))
        np.random.shuffle(self.train_index)
        self.train_data = self.train[self.train_index, :]
        self.batchsize = batchsize
        self.num = 0

    def batch_size(self, resize):
        self.num += self.batchsize
        return np.reshape(self.train_data[self.num - self.batchsize:self.num], resize)


def build( data, name, filter_size):
    with tf.name_scope(name) as scope:
        input = data
        input_shape = [-1, int(input.shape[1]), int(input.shape[2]), int(input.shape[3])]
        c1 = conv2d(input, name + 'c1', filter_size)
        p1 = maxpool2d(c1, name + 'p1')

        dc1 = deconv2d(p1, name + 'dc1', kshape=filter_size[:2], n_outputs=3)
        up1 = upsample(dc1, name + 'up1', factor=[2, 2])

        output = fullyConnected(up1, name + 'output', output_size=input.shape[1] * input.shape[2] * input.shape[3])
        with tf.name_scope(name + 'cost'):
            cost = tf.reduce_mean(
                tf.square(tf.subtract(tf.reshape(output, input_shape), data)))
        return c1, p1, dc1, up1, cost


def conv2d(input, name, kshape, strides=[1, 1, 1, 1]):
        with tf.variable_scope(name):
            W = tf.get_variable(name='w_' + name,
                                shape=kshape,
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b = tf.get_variable(name='b_' + name,
                                shape=[kshape[3]],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            out = tf.nn.conv2d(input, W, strides=strides, padding='SAME')
            out = tf.nn.bias_add(out, b)
            out = tf.nn.leaky_relu(out)
            return out

def deconv2d(input, name, kshape, n_outputs, strides=[1, 1]):
        with tf.variable_scope(name):
            out = tf.contrib.layers.conv2d_transpose(input,
                                                     num_outputs=n_outputs,
                                                     kernel_size=kshape,
                                                     stride=strides,
                                                     padding='SAME',
                                                     weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(
                                                         uniform=False),
                                                     biases_initializer=tf.contrib.layers.xavier_initializer(
                                                         uniform=False),
                                                     activation_fn=tf.nn.leaky_relu)
            return out

def maxpool2d(x, name, kshape=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
        with tf.variable_scope(name):
            out = tf.nn.max_pool(x,
                                 ksize=kshape,  # size of window
                                 strides=strides,
                                 padding='SAME')

            return out

def upsample(input, name, factor=[2, 2]):

        size = [int(input.shape[1]) * factor[0], int(input.shape[2]) * factor[1]]
        with tf.variable_scope(name):
            out = tf.image.resize_bilinear(input, size=size, align_corners=None, name=None)

            return out

def fullyConnected(input, name, output_size):
        with tf.variable_scope(name):
            input_size = input.shape[1:]
            input_size = int(np.prod(input_size))  # get total num of cells in one input image
            W = tf.get_variable(name='w_' + name,
                                shape=[input_size, output_size],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b = tf.get_variable(name='b_' + name,
                                shape=[output_size],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            input = tf.reshape(input, [-1, input_size])
            out = tf.nn.leaky_relu(tf.add(tf.matmul(input, W), b))

            return out

def test():
    data = Data('../data/data.npz', 64)
    x = tf.placeholder(tf.float32, [None, 32, 128, 3])
    with tf.Session() as sess:
        _, _, _, _, cost = build(x, 'buile_model', [5, 5, 3, 64])
        opt = tf.reduce_mean(cost)
        sess.run(tf.global_variables_initializer())
        print(len(data.train_data))

        for epoch in range(20):
            avg_cost = 0
            total_batch = int(len(data.train_data) / 64)
            data.num = 0
            for i in range(total_batch):
                cost1, _ = sess.run((cost, opt), feed_dict={x: data.batch_size([-1, 32, 128, 3])})
                avg_cost += cost1 / len(data.train_data) * 64

            print("Epoch:{},Cost:{:.9f}".format(epoch, avg_cost))

    '''with tf.Session() as sess:
        x = tf.placeholder(tf.float32, [None, 32, 128, 3])
        _,_,_,_,cost = build(x, 'buile_model',[5,5,3,64])
        _, _, _, _,cost1 = build(x, 'buile_model1',[5,5,3,64])
        sess.run(tf.global_variables_initializer())

        asd = sess.run(cost1,feed_dict={x:np.reshape(np.load('../data/data.npz')['train'][:64],[-1,32,128,3])})
        print(asd)'''



if __name__ == '__main__':
    test()