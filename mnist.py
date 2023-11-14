import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist

model_file_path = "digit_detect_ai.keras"

# Load saved model
loaded_model = tf.keras.models.load_model(model_file_path)

#Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Randomly select some images from the test set
num_samples = 5
random_indices = np.random.randint(0, X_test.shape[0], num_samples)
sample_images = X_test[random_indices]
sample_labels = y_test[random_indices]

#Image resize
sample_images = sample_images.reshape(sample_images.shape + (1,))

# Make predictions using the loaded model
predictions = loaded_model.predict(sample_images)

# Display the sample images and their predicted labels
for i in range(num_samples):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}\nActual: {sample_labels[i]}")
    plt.axis('off')

plt.show()
