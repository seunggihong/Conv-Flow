import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

cfg = {
    'A': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGGnet(Model):
    def __init__(self, input=None):
        super().__init__()

    def create_nets(self, config):
        nets = []
        for i in config:
            if i == 'M':
                nets.append(layers.MaxPool2D())
                continue
            nets.append(layers.Conv2D(
                i, kernel_size=(3, 3), activation='relu'))
            nets.append(layers.BatchNormalization())
            nets.append(layers.ReLU())

        return Sequential(nets)


if __name__ == "__main__":
    vgg = VGGnet()
    vgg.create_nets(cfg['A'])
    vgg.summary()
