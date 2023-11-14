from statistics import mode
import tensorflow as tf
from tensorflow.keras import layers, models

cfg = {
    'A': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def create_layers(config, num_classes, input_shape=(32, 32, 3)):
    model = models.Sequential()
    for i in config:
        if i == "M":
            model.add(layers.MaxPool2D())
            continue
        model.add(layers.Conv2D(i, kernel_size=(3, 3),
                  input_shape=input_shape, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


if __name__ == "__main__":
    vgg_A = create_layers(cfg['A'], 10)
    vgg_A.summary()
