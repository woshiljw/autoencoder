import tensorflow as tf


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
        self.pool = tf.nn.max_pool(self.conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        # the decoder layer

        self.unpool = hidden_transfer(
            tf.nn.conv2d_transpose(self.pool, self.weights['w2'],
                                   self.conv.get_shape(),
                                   [1, 2, 2, 1], padding='SAME') + self.weights['b2']
        )
        self.deconv = tf.nn.conv2d(
            self.unpool, self.weights['w3'], [1, 1, 1, 1], padding='SAME') + self.weights['b3']

        print('deconv: ',self.deconv.get_shape())
        self.cost = tf.reduce_mean(
            tf.square(self.x - self.deconv)
        )
        self.opt = tf.train.AdamOptimizer(0.001).minimize(self.cost)

        '''
        
        output_shape = [int(self.pool.get_shape()[0]),
                        int(self.pool.get_shape()[1]), int(self.pool.get_shape()[2]),
                        self.input_shape[3]]
        self.deconv = hidden_transfer(
            tf.nn.conv2d_transpose(self.pool, self.weights['w2'], output_shape, [1, 1, 1, 1], padding='SAME')
            + self.weights['b2']
        )
        self.unpool = tf.nn.conv2d_transpose(self.deconv, self.weights['w3'], self.x.get_shape(), [1, 2, 2, 1],
                                             padding='SAME')
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
            tf.truncated_normal([2, 2, self.filter_size[3], self.filter_size[3]], stddev=0.1))
        all_weights['b2'] = tf.Variable(tf.constant(0.1, shape=[self.filter_size[3]]))
        all_weights['w3'] = tf.Variable(
            tf.truncated_normal([self.filter_size[0], self.filter_size[1],
                                 self.filter_size[3], self.filter_size[2]], stddev=0.1)
        )
        print(all_weights['w3'].shape)
        all_weights['b3'] = tf.Variable(tf.constant(0.1, shape=[self.filter_size[2]]))

        # deconv_filter_size = [self.filter_size[0],self.filter_size[1],self.filter_size[3],self.filter_size[3]]

        '''
        
        all_weights['w2'] = tf.Variable(
            tf.truncated_normal(self.filter_size, stddev=0.1)
        )
        all_weights['b2'] = tf.Variable(
            tf.constant(0.1, shape=[self.filter_size[2]])
        )
        print([2, 2, self.input_shape[3], self.input_shape[3]])
        all_weights['w3'] = tf.Variable(
            tf.truncated_normal([2, 2, self.input_shape[3], self.input_shape[3]])
        )
        
        '''
        return all_weights

    @property
    def partial_fit(self):
        return (self.cost,self.opt)

    def calc_total_cost(self):
        return self.cost

    def reconstruct(self):
        return self.unpool


def test():
    import numpy as np

    data = np.load('../data/data.npz')
    with tf.Session() as sess:
        sae = Stacked_AutoEncoder([5, 5, 3, 64], [64, 32, 128, 3])
        sess.run(tf.global_variables_initializer())

        print(sess.run(sae.cost, feed_dict={sae.x: np.reshape(data['train'][:64], [-1, 32, 128, 3])}))
    '''
    
    
    print(data['train'].shape)
    '''


if __name__ == '__main__':
    test()
