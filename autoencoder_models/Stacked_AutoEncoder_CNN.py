import tensorflow as tf
from tensorflow.contrib.layers import batch_norm

class Stacked_AutoEncoder(object):
    def __init__(self, filter_size, input_shape,
                 hidden_transfer=tf.nn.relu):
        self.filter_size = filter_size
        self.x = tf.placeholder(tf.float32, input_shape)
        self.input_shape = input_shape
        self.weights = self._initialize_weights()

        # the encoder layer
        self.conv = hidden_transfer(
            tf.nn.conv2d(self.x, self.weights['w1'], [1, 1, 1, 1], padding="SAME")
            + self.weights['b1']
        )
        self.conv = batch_norm(self.conv)
        self.pool = tf.nn.max_pool(self.conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        # the decoder layer


        print(self.pool.get_shape())

        self.unpool = tf.image.resize_nearest_neighbor(self.pool, [32, 128])
        self.unpool = batch_norm(self.unpool)
        self.decode = hidden_transfer(tf.nn.conv2d_transpose(self.unpool,self.weights['w2'],[64,32,128,3],
                                             [1,1,1,1],padding="SAME"))

        self.decode = tf.sigmoid(
            tf.nn.conv2d(self.decode,
                         tf.truncated_normal([1,1,3,3]),[1,1,1,1],padding='SAME'
                         )
        )

        self.out = self.decode




        self.cost = tf.reduce_mean(
            tf.square(self.out-self.x)
        )
        self.opt = tf.train.AdamOptimizer().minimize(self.cost)

        '''
        
        '''

    def _initialize_weights(self):
        all_weights = dict()

        all_weights['w1'] = tf.Variable(
            tf.truncated_normal(self.filter_size, stddev=0.1)
        )
        all_weights['b1'] = tf.Variable(
            tf.constant(0.1, shape=[self.filter_size[3]])
        )
        all_weights['w2'] = tf.Variable(
            tf.truncated_normal([5,5,3,64], stddev=0.1)
        )
        all_weights['b2'] = tf.Variable(
            tf.constant(0.1, shape=[3])
        )
        '''
        '''
        return all_weights

    @property
    def partial_fit(self):
        return (self.cost,self.opt)

    def calc_total_cost(self):
        return self.cost

    def reconstruct(self):
        return self.out


def test():
    import numpy as np

    data = np.load('../data/data.npz')
    with tf.Session() as sess:
        sae = Stacked_AutoEncoder([5, 5, 3, 64], [64, 32, 128, 3])
        sess.run(tf.global_variables_initializer())
        out = sess.run(sae.cost, feed_dict={sae.x: np.reshape(data['train'][:64], [-1, 32, 128, 3])})

        print(out)
    '''
    
    
    print(data['train'].shape)
    '''


if __name__ == '__main__':
    test()
