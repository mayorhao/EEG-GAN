import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from Generators import Generator
from Discriminators import Discriminator

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

total_epoch = 150
batch_size = 100
learning_rate = 0.0002

n_input = 28 * 28
n_noise = 128  

def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))

X = tf.placeholder(tf.float32, [None, n_input])
Z = tf.placeholder(tf.float32, [None, n_noise])

G = Generator(n_input, n_noise)
# G.net(Z) returns fake generated sample from a random vector Z
G_out = G.net(Z)

D = Discriminator(n_input, n_noise)
# Realistic score on fake sample
D_gene = D.net(G_out, reuse = False)
# Realistic score on real sample
D_real = D.net(X, reuse = True)

# Define the different loss functions for both the generator and the discriminator
loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))
loss_G = tf.reduce_mean(tf.log(D_gene))

D_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
G_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')

train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D,
                                                         var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G,
                                                         var_list=G_var_list)
# Get ready to run the graph (run operations + evaluate variables)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples/batch_size)
loss_val_D, loss_val_G = 0, 0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise)

        _, loss_val_D = sess.run([train_D, loss_D],
                                 feed_dict={X: batch_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G],
                                 feed_dict={Z: noise})

    print('Epoch:', '%04d' % epoch,
          'D loss: {:.4}'.format(loss_val_D),
          'G loss: {:.4}'.format(loss_val_G))

    if epoch == 0 or (epoch + 1) % 10 == 0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(G_out, feed_dict={Z: noise})

        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))

        plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)

print('Complite')
