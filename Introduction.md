<h1>DigitDetect</h1>
One of the most prominent artificial intelligence applications are in the field of computer vision such as object detection and localization, visual data augmentation, visual navigation, and even in autonomous driving.

As my first stepping stone towards my independent research in computer vision, I developed DigitDetectAI, a RL trained model that takes in user input of a digit ranging from 1-10 and predicts what the user input digit is.

The original development of the project predicted 5 MNIST images at random, with a follow-up development where the model predicted user input images from an empty canvas.

The model was trained with the MNIST data using a CNN model in a **TinyVCG** architecture composed of 4 Conv2D layers, 2 MaxPool2D layers, 1 Flatten and Dense layers each. The model was trained in 10 epochs with an overall accuracy of **99.04%**.

You can download the model and implement your own ideas to train the model with other datasets that can be found [here](https://archive.ics.uci.edu/)!

mnist.py will demonstrate the model in action, randomly choosing MNIST digits and showing predictions. A more practical approach where users can input their own digits on a black canvas and predict using my trained model can be checked out on my website [here](https://sungwookm.github.io/projects/digitdetectai/)!
