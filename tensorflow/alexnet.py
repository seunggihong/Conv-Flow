from tensorflow.keras import layers, models, datasets
import tensorflow as tf
import matplotlib.pyplot as plt

# load CIFAR-10 dataset
# cifar10 = datasets.cifar10
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

# dispersion learning settings
strategy = tf.distribute.MirroredStrategy()

# data input pipeline


# Alexnet model
with strategy.scope():
    model = models.Sequential(
        [
            layers.Conv2D(filters=48, kernel_size=(11, 11),
                          strides=(4, 4), input_shape=(224, 224, 3), activation='relu'),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

            layers.Conv2D(filters=128, kernel_size=(5, 5),
                          strides=(1, 1), activation='relu', padding='same'),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

            layers.Conv2D(filters=192, kernel_size=(3, 3),
                          strides=(1, 1), activation='relu', padding='same'),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

            layers.Conv2D(filters=192, kernel_size=(3, 3),
                          strides=(1, 1), activation='relu', padding='same'),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

            layers.Conv2D(filters=128, kernel_size=(3, 3),
                          strides=(1, 1), activation='relu', padding='same'),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

            layers.Flatten(),
            layers.Dense(2048, activation='relu'),
            layers.Dense(2048, activation='relu'),
            layers.Dense(10, activation='softmax')
        ]
    )
