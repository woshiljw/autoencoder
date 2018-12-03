from autoencoder_models.DenoisingAutoencoder import AdditiveGaussianNoiseAutoencoder

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('./data',one_hot=True)

def standard_scale(X_train,X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train,X_test

def get_random_black_from_data(data,batch_size):
    start_index = np.random.randint(0,len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]

X_train,X_text = standard_scale(mnist.train.images,mnist.test.images)

n_samples = int(mnist.train.num_examples)

train_epochs = 20

batch_sizd = 128

desplay_step = 10

gae = AdditiveGaussianNoiseAutoencoder(
    n_input=784,
    n_hidden=200,
    transfer_function=tf.nn.softplus,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
    scale=0.1
)

for epoch in range(train_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_sizd)

    for i in range(total_batch):
        batch_xs = get_random_black_from_data(X_train, batch_sizd)

        cost = gae.partial_fit(batch_xs)

        avg_cost += cost / n_samples * batch_sizd

    if epoch % desplay_step == 0:
        print("Epoch:", '%d,' % (epoch + 1), "Cost:", "{:.9f}".format(avg_cost))

print("Total cost: " + str(gae.calc_total_cost(X_text)))
plt.imshow(np.reshape(gae.reconstruct(X_text),(-1,28,28))[0])
plt.show()