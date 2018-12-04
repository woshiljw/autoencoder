import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
class Autoencoder(object):

    def __init__(self,filter_size,input_shape,
                 hidden_transfer_function=tf.nn.relu,
                 output_transfer_function=tf.nn.relu,
                 optimizer=tf.train.AdamOptimizer()):
        self.filter_size = filter_size
        self.input_shape = input_shape
        self.weights = self._initialize_weights()
        self.hidden_transfer = hidden_transfer_function
        self.output_transfer = output_transfer_function

        self.x = tf.placeholder(tf.float32,shape=self.input_shape)
        self.hidden_conv = self.hidden_transfer(
            tf.nn.conv2d(self.x,self.weights['w1'],
                         [1,1,1,1],padding='SAME')+self.weights['b1']
        )
        self.hidden_pool = tf.nn.max_pool(self.hidden_conv,[1,2,2,1],
                                          [1,2,2,1],padding='SAME')

        self.unpool = self.hidden_transfer(
            tf.nn.conv2d_transpose(self.hidden_pool,self.weights['w2'],
                                   self.hidden_conv.get_shape(),
                                   [1,2,2,1],padding='SAME')+self.weights['b2']
        )
        self.deconv = tf.nn.conv2d(
                self.unpool,self.weights['w3'],[1,1,1,1],padding='SAME')+self.weights['b3']

        self.cost = tf.reduce_mean(
            tf.pow(
                tf.subtract(self.deconv,self.x),2.0
            )
        )

        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
    def _initialize_weights(self):
        all_weights = dict()

        all_weights['w1'] = tf.Variable(tf.truncated_normal(self.filter_size))
        all_weights['b1'] = tf.Variable(tf.constant(0.1,shape=[self.filter_size[3]]))
        all_weights['w2'] = tf.Variable(tf.truncated_normal([2,2,self.filter_size[3],self.filter_size[3]],stddev=0.1))
        all_weights['b2'] = tf.Variable(tf.constant(0.1,shape=[self.filter_size[3]]))
        all_weights['w3'] = tf.Variable(
            tf.truncated_normal([self.filter_size[0],self.filter_size[1],
                                 self.filter_size[3],self.filter_size[2]],stddev=0.1)
        )
        all_weights['b3'] = tf.Variable(tf.constant(0.1,shape=[self.filter_size[2]]))

        return all_weights

    def test_fit(self,X):
        return self.sess.run(self.deconv,feed_dict={self.x:X})

    def partial_fit(self,X):
        cost,opt = self.sess.run((self.cost,self.optimizer),feed_dict={self.x:X})
        return cost

    def calc_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict={self.x:X})

    def transform(self,X):
        return self.sess.run(self.hidden_pool,feed_dict={self.x:X})

    def generate(self,hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['encode'][-1]['b'].shape)
        return self.sess.run(self.deconv,feed_dict={self.hidden_pool:hidden})

    def reconstruct(self,X):
        return self.sess.run(self.deconv,feed_dict={self.x:X})

def test():
    mnist = input_data.read_data_sets('../data',one_hot=True)
    autoencoder = Autoencoder([3,3,1,36],
                              [64,28,28,1])
    out = autoencoder.test_fit(np.reshape(mnist.test.images[:64],[-1,28,28,1]))
    print(np.shape(out))


if __name__ == '__main__':
    test()