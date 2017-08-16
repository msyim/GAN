import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt

# Data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# HyperParameters
batch_size = 256
z_size = 128
h1_size = 256
h2_size = 512
epochs = 500
K = 1

X = tf.placeholder(tf.float32, shape=[None, 784])
keep_prob = tf.placeholder(tf.float32)

def weight_variable(name, shape):
  #return tf.get_variable(name, shape=shape, initializer=tf.truncated_normal_initializer())
  return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

weights = {
  "DW1" : weight_variable("DW1", [784,h2_size]),
  "Db1" : weight_variable("Db1", [h2_size]),
  "DW2" : weight_variable("DW2", [h2_size,h1_size]),
  "Db2" : weight_variable("Db2", [h1_size]),
  "DW3" : weight_variable("DW3", [h1_size,1]),
  "Db3" : weight_variable("Db4", [1]),

  "GW1" : weight_variable("GW1", [z_size,h1_size]),
  "Gb1" : weight_variable("Gb1", [h1_size]),
  "GW2" : weight_variable("GW2", [h1_size,h2_size]),
  "Gb2" : weight_variable("Gb2", [h2_size]),
  "GW3" : weight_variable("GW3", [h2_size,784]),
  "Gb3" : weight_variable("Gb3", [784])
}

def Discriminator(data_merged):
  hidden1 = tf.nn.dropout(tf.nn.relu(tf.matmul(data_merged, weights["DW1"]) + weights["Db1"]), keep_prob)
  hidden2 = tf.nn.dropout(tf.nn.relu(tf.matmul(hidden1, weights["DW2"]) + weights["Db2"]), keep_prob)
  hidden3 = tf.nn.sigmoid(tf.matmul(hidden2, weights["DW3"]) + weights["Db3"])
 
  return hidden3

def Generator(data_z):
  # Starts from a Gaussian random with mean = 0, stdv = 0.01 of size 128
  hidden1 = tf.nn.relu(tf.matmul(data_z, weights["GW1"]) + weights["Gb1"])
  hidden2 = tf.nn.relu(tf.matmul(hidden1, weights["GW2"]) + weights["Gb2"])
  hidden3 = tf.nn.tanh(tf.matmul(hidden2, weights["GW3"]) + weights["Gb3"])

  return hidden3

data_z = tf.truncated_normal(shape=[batch_size,z_size])
data_g = Generator(data_z)

pred = Discriminator(X)
fake_pred = Discriminator(data_g)

generator_var_list = [weights[i] for i in ["GW1","Gb1","GW2","Gb2","GW3","Gb3"]]
discriminator_var_list= [weights[i] for i in ["DW1","Db1","DW2","Db2","DW3","Db3"]]

# In section 3 of the paper, the author suggests maximizing "log(D(G(z)))" 
# instead of minimizing "log(1-D(G(z)))" because "log(1-D(G(z)))" saturates when G is
# not good enough at counterfeiting.
initCriterion = (-1)*tf.reduce_mean(tf.log(fake_pred))
initOptim = tf.train.AdamOptimizer(0.0001).minimize(initCriterion, var_list=generator_var_list)

# After G learns for a bit (seems like about 2 epochs should be enough) we start using
# the loss function for G and D defined in the algorithm.
stableCriterionForD = (-1)*tf.reduce_mean(tf.log(pred) + tf.log(1-fake_pred))
stableOptForD = tf.train.AdamOptimizer(0.0001).minimize(stableCriterionForD, var_list=discriminator_var_list)
stableCriterionForG = tf.reduce_mean(tf.log(1-fake_pred))
stableOptForG = tf.train.AdamOptimizer(0.0001).minimize(stableCriterionForG, var_list=generator_var_list)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for epoch in range(epochs):
    num_batches = int(mnist.train.num_examples/(K*batch_size))
    for i in range(num_batches):
      # training ratio D:G = K:1 
      # First train D for K times
      for index in range(K):
        batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x = 2*batch_x - 1
        dloss, _, p,fp = sess.run([stableCriterionForD, stableOptForD, pred, fake_pred], feed_dict={X:batch_x, keep_prob: 1})

      # use initial criterion for about 2 epochs
      if epoch < 2: 
        gloss,_,fp,g,z = sess.run([initCriterion, initOptim, fake_pred,data_g,data_z], feed_dict={keep_prob:1.0})
        print "[E:%d][B:%d] gloss : %f" % (epoch,i,gloss)

        # show samples of generated images.
        for j in range(0, 3):
          plt.subplot(311 + (j))
          image = np.reshape(g[j],[28,28])
          plt.imshow(image, interpolation='nearest', cmap='gray')
        plt.show()
      
      # After 2 epochs, use stable criterion
      else :
        gloss,_,fp,g = sess.run([stableCriterionForG, stableOptForG,fake_pred,data_g], feed_dict={keep_prob:1})

    print "[E: %d] dloss: %f, gloss: %f" %(epoch, dloss, gloss)
    for j in range(0,9):
      plt.subplot(331 + (j))
      image = np.reshape(g[j],[28,28])
        
      plt.imshow(image, interpolation='nearest', cmap='gray')
    plt.show()
