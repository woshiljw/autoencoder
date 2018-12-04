import tensorflow as tf
from autoencoder_models.Autoencoder_CNN import Autoencoder
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./data',one_hot=True)

ae1 = Autoencoder(filter_size=[5,5,1,32],
                  input_shape=[64,28,28,1])
ae2 = Autoencoder(filter_size=[3,3,32,64],
                  input_shape=[64,14,14,32])

x = tf.placeholder(tf.float32,[None,28,28,1])
