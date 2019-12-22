import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# creating the model
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28,28)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(10, activation='softmax')
# ])

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

## training the model
# model.fit(train_images, train_labels, epochs=20)

# # saving the model
# model.save('saved_models/fashion_mnist_nn_model.h5')
#
# # making predictions with the model
# predictions = model.predict(test_images)

# loading the saved model
saved_model = tf.keras.models.load_model('saved_models/fashion_mnist_nn_model.h5')

# making predictions with the saved model
predictions = saved_model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("What it actually is: " + classe_names[test_labels[i]])
    plt.title("Neural network predicted: " + class_names[np.argmax(predictions[i])])
    plt.show()