import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from autoencoder_models.Autoencoder_CNN import Autoencoder

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

train_epochs = 10

batch_sizd = 64

desplay_step = 1

autoencoder = Autoencoder(optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                          input_shape=[64,28,28,1],
                          filter_size=[3,3,1,36])

for epoch in range(train_epochs):
    avg_cost = 0
    total_batch = int(n_samples/batch_sizd)

    for i in range(total_batch):
        batch_xs = get_random_black_from_data(X_train,batch_sizd)



        cost = autoencoder.partial_fit(np.reshape(batch_xs,(-1,28,28,1)))

        avg_cost += cost / n_samples * batch_sizd

    if epoch%desplay_step == 0:
        print("Epoch:",'%d,'%(epoch+1),"Cost:","{:.9f}".format(avg_cost))

print("Total cost: " + str(autoencoder.calc_total_cost(np.reshape(X_text,(-1,28,28,1)))))