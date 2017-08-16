# GAN
An implementation of the paper : "Generative Adversarial Nets" using TensorFlow.  Impelemented initial training criterion (maximizing log(D(G(z)))) as suggested in the paper in addition to the loss functions defined in the main algorithm.

# Architecture Summary

* Generator layers:
 1) 128-d random Gaussian 
 2) 256-d fully connected/relu
 3) 512-d fully connected/relu
 4) 784-d fully connected/tanh

* Discriminator layers:
 1) 784-d MNIST pixel values
 2) 512-d fully connected/relu
 3) 256-d fully connected/relu
 4) 128-d fully connected/relu
 5) 1-d   fully connected/sigmoid

# Generated images after training 500 epochs
