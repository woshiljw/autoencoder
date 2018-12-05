import numpy as np
import tensorflow as tf

class Data(object):
    def __init__(self,datafile,batchsize):
        self.img = np.load(datafile)['train']
        self.train = self.img[4480:]
        self.train_index = np.arange(len(self.train))
        np.random.shuffle(self.train_index)
        self.train_data = self.train[self.train_index,:,:,:]
        self.num = 0
    def batch_size(self,size,resize):
        self.num += size
        return np.reshape(self.train_data[self.num-size:self.num],resize)

