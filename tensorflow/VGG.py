import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

cfg = {
    'A': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGGnet(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_nets(self, config):
        nets = []
        for i in config:
            if i == 'M':
                nets.append(layers.MaxPool2D())
            nets.append(layers.Conv2D())
            nets.append(layers.BatchNormalization())
            nets.append(layers.ReLU())

        return Sequential(nets)
