<h1>Project Description</h1>

To train the model, we will use the [MNIST dataset](https://www.tensorflow.org/datasets/catalog/mnist).
In a MNIST dataset, each of the 60,000 digits are stored as a numbered array.

<img src='/images/mnist_array.png' width='300' height='300'>

For the purpose of developing this model, MNIST data was recalibrated to size 28 x 28 and grayscale for training the model.

<img src='/images/mnist_recalibration.png' width='800' height='300'>

The data was then trained with 8 layers as shown in the image below.

<img src='/images/mnist_array.png' width='700' height='300'>

With 10 epochs, we are able to train a model with an overall accuracy of **98.91%**.
