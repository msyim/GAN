# GAN
An implementation of the paper : "Generative Adversarial Nets" using TensorFlow.  Impelemented initial training criterion (maximizing log(D(G(z)))) as suggested in the paper in addition to the loss functions defined in the main algorithm.

### Architecture Summary

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

### Generated images after training with initial criterion for 2 epochs
* After Epoch 1

![](https://github.com/msyim/GAN/blob/master/images/GANepoch1.png)

* After Epoch 2

![](https://github.com/msyim/GAN/blob/master/images/GANepoch2.png)

### Generated images after training 500 epochs (in addition to the training 2 epochs with initial criterion)

* After Epoch 10

![](https://github.com/msyim/GAN/blob/master/images/GANepoch10.png)

* After Epoch 30

![](https://github.com/msyim/GAN/blob/master/images/GANepoch30.png)

* After Epoch 100

![](https://github.com/msyim/GAN/blob/master/images/GAN100.png)


* After Epoch 200

![](https://github.com/msyim/GAN/blob/master/images/GAN200.png)


* After Epoch 300

![](https://github.com/msyim/GAN/blob/master/images/GAN300.png)


* After Epoch 400

![](https://github.com/msyim/GAN/blob/master/images/GAN400.png)


* After Epoch 500

![](https://github.com/msyim/GAN/blob/master/images/GAN500.png)
