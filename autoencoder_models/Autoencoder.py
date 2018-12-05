from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class Autoencoder(object):

    def __init__(self,n_layers,transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer()):

        self.n_layers = n_layers
        self.transfer = transfer_function

        network_weights = self._initialize_weights()
        self.weights = network_weights

        self.x = tf.placeholder(tf.float32,[None,self.n_layers[0]])
        self.hidden_encode = []
        h = self.x

        for layer in range(len(self.n_layers)-1):
            #print(self.weights['encode'][layer])
            h = self.transfer(
                tf.add(
                    tf.matmul(h,self.weights['encode'][layer]['w']),
                    self.weights['encode'][layer]['b']
                )
            )
            self.hidden_encode.append(h)

        self.hidden_recon = []

        for layer in range(len(self.n_layers)-1):
            h = self.transfer(
                tf.add(
                    tf.matmul(h,self.weights['recon'][layer]['w']),
                    self.weights['recon'][layer]['b']
                )
            )
            self.hidden_recon.append(h)

        self.reconstruction = self.hidden_recon[-1]

        self.cost = 0.5*tf.reduce_mean(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))
        self.optimizer = optimizer.minimize(self.cost)

    def _initialize_weights(self):
        all_weight = dict()
        initializer = tf.contrib.layers.xavier_initializer()

        encoder_weights = []
        for layer in range(len(self.n_layers)-1):
            w = tf.Variable(
                initializer((self.n_layers[layer],self.n_layers[layer+1]),
                            dtype=tf.float32))
            b = tf.Variable(
                tf.zeros([self.n_layers[layer+1]],dtype=tf.float32))
            encoder_weights.append({'w':w,'b':b})

        recon_weights = []

        for layer in range(len(self.n_layers)-1,0,-1):
            w = tf.Variable(
                initializer((self.n_layers[layer], self.n_layers[layer - 1]),
                            dtype=tf.float32))
            b = tf.Variable(
                tf.zeros([self.n_layers[layer - 1]], dtype=tf.float32))
            recon_weights.append({'w': w, 'b': b})

        all_weight['encode'] = encoder_weights
        all_weight['recon'] = recon_weights
        return all_weight

    def partial_fit(self):
        return (self.cost,self.optimizer)

    def calc_total_cost(self,X):
        return self.cost

    def transform(self):
        return self.hidden_encode[-1]

    def generate(self,hidden=None):
        return self.reconstruction

    def reconstruct(self,X):
        return self.reconstruction
    def getWeights(self):
        #raise NotImplementedError
        return self.weights

    def getBias(self):
        return self.weights