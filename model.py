import random
import math
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import layers
from keras.datasets import mnist
from keras.layers import Dense

(X_train, y_train), (X_test, y_test) = mnist.load_data() #Load data

#Dataset preset
X_train = X_train.reshape(X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))

X_train = X_train / 255
X_test = X_test / 255

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

#CNN model
model = tf.keras.Sequential([
    layers.Conv2D(filters = 10,
                  kernel_size = 3,
                  activation = "relu",
                  input_shape = (28, 28, 1)),
    layers.Conv2D(10, 3, activation = "relu"),
    layers.MaxPool2D(),
    layers.Conv2D(10, 3, activation = "relu"),
    layers.Conv2D(10, 3, activation = "relu"),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(10, activation = "softmax")
])

model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = tf.keras.optimizers.legacy.Adam(),
              metrics = ["accuracy"]
              )

model.fit(X_train, y_train, epochs = 10)

model.evaluate(X_test, y_test)

model.save("digit_detect_ai.keras")
