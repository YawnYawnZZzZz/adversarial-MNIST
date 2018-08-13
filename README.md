# adversarial-MNIST
Create adversarial images to fool a MNIST classifier in TensorFlow

The code is in [main.py](https://github.com/YawnYawnZZzZz/adversarial-MNIST/blob/master/main.py). The relevant plots are in [plots](https://github.com/YawnYawnZZzZz/adversarial-MNIST/tree/master/plots). There are comments to each item in commit history.

### Steps taken:

1. We start with sample code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_deep.py. We aim to create adversarial images of '2' so the classifier thinks they are '6'.

1. We would like to apply 'fast gradient sign method' (see [Goodfellow](https://arxiv.org/abs/1412.6572)) after training the network. Compare the following two ways: (Ah! I put 'up' and 'down' the wrong way round in the comments in code.)
    1. Step up the hill in y=2 landscape, and step down the hill in y=6 landscape.
    2. Step down the hill in y=6 landscape only.
    
    From the examples generated, it turned out it is sufficient to step downhill in y=6 landscape (ii). But in both cases, the model is fooled to some extent.

1. Apply 'fast gradient sign method' and modify the input image `x` by updating it to be to
 `-epsilon * sign(d(loss)/dx) + x`, where `epsilon` is a small number similar to the learning rate, `sign` is the sign function, and `loss` is the loss function when training the neural network, pretending the label of image is '6'.
    
    We also reduce number of steps (batches) from 20000 to 2000, just to get a rough idea. It takes too long to have 20000 batches.

1. When the number of batches is 200, one step downhill with `epsilon=0.1` fools the model well. But when the number of batches is 2000, taking just one step downhill is not sufficient for the model to be confident that the new image is a '6'. We take `epochs` many steps. Experiments shows `epochs` around 10 works well. (See below.)

1. We then plot 10 random samples from the MNIST.test images. Each sample corresponds to one row, three columns. 
    1. Column 1 shows the original image, its predicted label, and the confidence (probability) of the model in that label.
    2. Column 2 shows the `delta`, i.e. the difference between the original image and the modified image. Note that delta is not scaled to the range [0,1] (as opposed to the case for the new image). But it gives a sense of the difference, which seems like noice but it is not.
    3. Column 3 shows the new image, and the confidence (probability) of the model in the image being a '6'.
    
### Some observations:
1. Note that even when the number of batches is only 201 and the overall test accuracy is low, the model is quite confident in classifying the modified images as 6. This may due to the fact that we are stepping downhill on the _current_ landscape, and it's the _current_ landscape where we calculate the corresponding probabilities.

2. It is astonishing how few steps (epochs) we need to fool the model so it has quite high confidence.

3. There are some problems associated with the Gradient Descent method. E.g. for taco-shaped landscapes, and at local minima and saddle points. Here we are using Gradient Ascent, which may have similar problems. Methods corresponding to improvements to Gradient Descent (e.g. Adam?) may give better adversarial results.

### Random thoughts:
1. May use another network to find optimal values of `epsilon` and `epochs`?
1. I naively thought that maybe one could first distinguish between a natural image and a modified adversarial image, to prevent malicious usage of adversarial examples. But Goodfellow mentioned in a talk that it is the posterior distribution `P(label=y|x)` over the class of labels y given imputs x that matters, rather than the distribution  `P(x)` of inputs x. (I may need to have a closer look at this.)
1. Now we are modifying an image at its input. May try modifying it at hidden layer?
1. [Goodfellow](https://arxiv.org/abs/1412.6572) says
    > adversarial examples occur in contiguous regions of the 1-D subspace defined by the fast gradient sign method
    
    Meanwhile there are results showing that training on adversarial examples well impoves the model on adversarial and natural examples. Is it the case that training on adversarial examples ensures the 'adversary' runs out of dimensions?

### Image results and Comments:
#### When the number of batches for model-training is 2001:
* When [epsilon=0.1 and epochs=1](https://github.com/YawnYawnZZzZz/adversarial-MNIST/blob/master/plots/2001%20eps%3D0.10_epochs%3D1_test_accu0.98.png), the model has low confidence that the modified images are 6. The number of steps may be too low. Overall test accuracy = 0.98. Modification Success rate = 0.16.

* When [epsilon=0.1 and epochs=3](https://github.com/YawnYawnZZzZz/adversarial-MNIST/blob/master/plots/2001%20eps%3D0.10_epochs%3D3_test_accu0.97.png), the model has some confidence that the modified images are 6. Overall test accuracy = 0.97. Modification Success rate = 0.98.

* When [epsilon=0.1 and epochs=5](https://github.com/YawnYawnZZzZz/adversarial-MNIST/blob/master/plots/2001%20eps%3D0.10_epochs%3D5_test_accu0.98.png), the model has high confidence that the modified images are 6. But the images are further from 2 from human eyes. Overall test accuracy = 0.98. Modification Success rate = 1.00.

* When [epsilon=0.05 and epochs=15](https://github.com/YawnYawnZZzZz/adversarial-MNIST/blob/master/plots/2001%20eps%3D0.05_epochs%3D15_test_accu0.97.png), the model has high confidence that the modified images are 6. But the images are further from 2 from human eyes. We have modified the image too much. Overall test accuracy = 0.97. Modification Success rate = 1.00.

* When [epsilon=0.01 and epochs=10](https://github.com/YawnYawnZZzZz/adversarial-MNIST/blob/master/plots/2001%20eps%3D0.01_epochs%3D10_test_accu0.98.png), the model has very low confidence that the modified images are 6. The steps (0.01) may be too small. Overall test accuracy = 0.98. Modification Success rate = 0.06.

* **[epsilon=0.05 and epochs=8](https://github.com/YawnYawnZZzZz/adversarial-MNIST/blob/master/plots/2001%20eps%3D0.05_epochs%3D8_test_accu0.98.png) seems to give a good balance of the level of confidence and the degree of modification.**

#### When the number of batches for model-training is 201:
* [epsilon=0.05 and epochs=10](https://github.com/YawnYawnZZzZz/adversarial-MNIST/blob/master/plots/201%20eps%3D0.05_epochs%3D10_test_accu0.91.png). Overall test accuracy = 0.98. Modification Success rate = 1.00.

