import numpy as np
import tensorflow as tf
from autoencoder_models.Stacked_AutoEncoder_CNN import Stacked_AutoEncoder


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


sae = Stacked_AutoEncoder([5, 5, 3, 64], [64, 32, 128, 3])

data = Data('./data/data.npz', 64)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(len(data.train_data))

    for epoch in range(20):
        avg_cost = 0
        total_batch = int(len(data.train_data) / 64)
        data.num=0
        for i in range(total_batch):
            cost, _ = sess.run(sae.partial_fit, feed_dict={sae.x: data.batch_size([-1, 32, 128, 3])})
            avg_cost +=cost/len(data.train_data)*64

        print("Epoch:{},Cost:{:.9f}".format(epoch,avg_cost))