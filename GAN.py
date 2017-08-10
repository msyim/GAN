import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# HyperParameters
batch_size = 64
z_size = 128
epochs = 10
K = 1

X = tf.placeholder(tf.float32, shape=[None, 784])

def weight_variable(name, shape):
  #return tf.get_variable(name, shape=shape, initializer=tf.truncated_normal_initializer())
  return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

weights = {
  "DW1" : weight_variable("DW1", [784,256]),
  "Db1" : weight_variable("Db1", [256]),
  "DW2" : weight_variable("DW2", [256,128]),
  "Db2" : weight_variable("Db2", [128]),
  "DW3" : weight_variable("DW3", [128,32]),
  "Db3" : weight_variable("Db3", [32]),
  "DW4" : weight_variable("DW4", [32,1]),
  "Db4" : weight_variable("Db4", [1]),

  "GW1" : weight_variable("GW1", [128,256]),
  "Gb1" : weight_variable("Gb1", [256]),
  "GW2" : weight_variable("GW2", [256,512]),
  "Gb2" : weight_variable("Gb2", [512]),
  "GW3" : weight_variable("GW3", [512,784]),
  "Gb3" : weight_variable("Gb3", [784])
}

def Discriminator(data_merged):
  hidden1 = tf.nn.relu(tf.matmul(data_merged, weights["DW1"]) + weights["Db1"])
  hidden2 = tf.nn.relu(tf.matmul(hidden1, weights["DW2"]) + weights["Db2"])
  hidden3 = tf.nn.relu(tf.matmul(hidden2, weights["DW3"]) + weights["Db3"])

  # Using sigmoid for the last layer since we're outputting the "probability"
  # that the data came from p_data
  hidden4 = tf.nn.sigmoid(tf.matmul(hidden3, weights["DW4"]) + weights["Db4"])
  return hidden4

def Generator(data_z):
  # Starts from a Gaussian random with mean = 0, stdv = 0.01 of size 128
  hidden1 = tf.nn.relu(tf.matmul(data_z, weights["GW1"]) + weights["Gb1"])
  hidden2 = tf.nn.relu(tf.matmul(hidden1, weights["GW2"]) + weights["Gb2"])

  # Using sigmoid for the last layer since MNIST data is in the range[0,1]
  hidden3 = tf.nn.sigmoid(tf.matmul(hidden2, weights["GW3"]) + weights["Gb3"])
  return hidden3

data_z = tf.truncated_normal(shape=[batch_size,z_size])
data_g = Generator(data_z)

pred = Discriminator(X)
fake_pred = Discriminator(data_g)

generator_var_list = [weights[i] for i in ["GW1","Gb1","GW2","Gb2","GW3","Gb3"]]
discriminator_var_list= [weights[i] for i in ["DW1","Db1","DW2","Db2","DW3","Db3","DW4","Db4"]]

initCriterion = tf.reduce_mean(tf.log(fake_pred))
initOptim = tf.train.AdamOptimizer(0.001).minimize((-1)*initCriterion, var_list=generator_var_list)

stableCriterionForD = tf.reduce_mean(tf.log(pred) + tf.log(1-fake_pred))
stableOptForD = tf.train.AdamOptimizer(0.001).minimize((-1)*stableCriterionForD, var_list=discriminator_var_list)

stableCriterionForG = tf.reduce_mean(tf.log(1-fake_pred))
stableOptForG = tf.train.AdamOptimizer(0.001).minimize(stableCriterionForG, var_list=generator_var_list)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for epoch in range(epochs):
    num_batches = int(mnist.train.num_examples/(K*batch_size))
    for i in range(num_batches):
      # training ratio D:G = K:1 
      # First train D for K times
      if epoch == 0 and i < 5 : 
        #gloss, _, fp = sess.run([initCriterion,initOptim,fake_pred]) 
        gloss,_,fp = sess.run([stableCriterionForG, stableOptForG,fake_pred])
        print "[E:%d][B:%d] gloss: %f" % (epoch,i,gloss)
        print fp
      else :
        for index in range(K):
          batch_x, _ = mnist.train.next_batch(batch_size)
          dloss, _, p,fp = sess.run([stableCriterionForD, stableOptForD, pred, fake_pred], feed_dict={X:batch_x})
          print "[E:%d][B:%d][K:%d] dloss: %f" % (epoch,i,index,dloss)
          print p
          print fp
        gloss,_,fp = sess.run([stableCriterionForG, stableOptForG,fake_pred])
        print "[E:%d][B:%d] gloss: %f" % (epoch,i,gloss)
        print fp
