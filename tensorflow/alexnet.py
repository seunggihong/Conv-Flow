from tensorflow.keras import layers, models, datasets
import tensorflow as tf
import matplotlib.pyplot as plt

cifar10 = datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

Alexnet = models.Sequential(
    [
        layers.experimental.preprocessing.Resizing(
            227, 227, interpolation='bilinear', input_shape=x_train.shape[1:]),

        layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(
            4, 4), input_shape=(224, 224, 3), activation='relu'),
        layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(
            1, 1), activation='relu', padding='same'),
        layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(
            1, 1), activation='relu', padding='same'),
        layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(
            1, 1), activation='relu', padding='same'),
        layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(
            1, 1), activation='relu', padding='same'),
        layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(.5),
        layers.Dense(10, activation='softmax')
    ]
)

Alexnet.compile(optimizer=tf.optimizers.SGD(learning_rate=0.001),
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])
Alexnet.summary()

Alexnet.fit(x_train, y_train, epochs=1)
