import json

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

n_z = 100


def conved_image_size(image_size, max_n_ch=512):
    n_conv = 4
    conved_size = image_size
    for i in range(n_conv):
        kwargs = get_conv2d_kwargs(i, image_size=image_size, max_n_ch=max_n_ch)
        conved_size = check_conved_size(conved_size, kwargs['ksize'], kwargs[
                                        'stride'], kwargs['pad'])
    return conved_size


def check_conved_size(image_size, ksize, stride=1, pad=0):
    return int((image_size - ksize + 2 * pad) / stride) + 1


def get_conv2d_kwargs(i, image_size=64, n_color=3, max_n_ch=512, deconv=False):
    n_conv = 4
    channels = [n_color] + [max_n_ch // (2 ** x)
                            for x in range(0, n_conv)][::-1]

    if image_size == 64:
        kernel = [
            {'ksize': 4, 'stride': 2, 'pad': 1},  # 64 -> 32
            {'ksize': 4, 'stride': 2, 'pad': 1},  # 32 -> 16
            {'ksize': 4, 'stride': 2, 'pad': 1},  # 16 -> 8
            {'ksize': 4, 'stride': 2, 'pad': 1},  # 8 -> 4
        ]
    elif image_size == 32:
        kernel = [
            {'ksize': 3, 'stride': 1, 'pad': 1},  # 32 -> 32
            {'ksize': 4, 'stride': 2, 'pad': 1},  # 32 -> 16
            {'ksize': 4, 'stride': 2, 'pad': 1},  # 16 -> 8
            {'ksize': 4, 'stride': 2, 'pad': 1},  # 8 -> 4
        ]
    elif image_size == 28:
        kernel = [
            {'ksize': 3, 'stride': 3, 'pad': 1},  # 28 -> 10
            {'ksize': 2, 'stride': 2, 'pad': 1},  # 10 -> 6
            {'ksize': 2, 'stride': 2, 'pad': 1},  # 6 -> 4
            {'ksize': 2, 'stride': 2, 'pad': 1},  # 4 -> 3
        ]
    else:
        raise NotImplementedError(
            '(image_size == {}) is not implemented'.format(image_size))

    if deconv:
        kwargs = {
            'in_channels': channels[n_conv - i],
            'out_channels': channels[n_conv - i - 1],
        }
        kwargs.update(kernel[n_conv - i - 1])
    else:
        kwargs = {
            'in_channels': channels[i],
            'out_channels': channels[i + 1],
        }
        kwargs.update(kernel[i])

    return kwargs


class Generator(chainer.Chain):

    def __init__(self, image_size, n_color, wscale=0.02):
        self.image_size = image_size
        self.n_color = n_color
        self.wscale = wscale
        self.conved_size = conved_image_size(image_size)
        super(Generator, self).__init__(
            l0=L.Linear(n_z, self.conved_size ** 2 * 512, wscale=wscale),
            dc1=L.Deconvolution2D(
                **get_conv2d_kwargs(0, image_size, n_color, 512, deconv=True), wscale=wscale),
            dc2=L.Deconvolution2D(
                **get_conv2d_kwargs(1, image_size, n_color, 512, deconv=True), wscale=wscale),
            dc3=L.Deconvolution2D(
                **get_conv2d_kwargs(2, image_size, n_color, 512, deconv=True), wscale=wscale),
            dc4=L.Deconvolution2D(
                **get_conv2d_kwargs(3, image_size, n_color, 512, deconv=True), wscale=wscale),
            bn0l=L.BatchNormalization(self.conved_size ** 2 * 512),
            bn0=L.BatchNormalization(512),
            bn1=L.BatchNormalization(256),
            bn2=L.BatchNormalization(128),
            bn3=L.BatchNormalization(64),
        )

    def __call__(self, z, test=False):
        h = self.l0(z)
        h = self.bn0l(h, test=test)
        h = F.relu(h)
        h = F.reshape(h, (z.data.shape[0], 512,
                          self.conved_size, self.conved_size))

        h = self.dc1(h)
        h = self.bn1(h, test=test)
        h = F.relu(h)

        h = self.dc2(h)
        h = self.bn2(h, test=test)
        h = F.relu(h)

        h = self.dc3(h)
        h = self.bn3(h, test=test)
        h = F.relu(h)

        x = self.dc4(h)

        return x

    def make_hidden(self, batchsize):
        return np.random.uniform(-1, 1, (batchsize, n_z, 1, 1)).astype(np.float32)

    def to_json(self):
        d = {
            'class_name': self.__class__.__name__,
            'kwargs': {
                'image_size': self.image_size,
                'n_color': self.n_color,
                'wscale': self.wscale,
            }
        }
        return json.dumps(d)


class Discriminator(chainer.Chain):

    def __init__(self, image_size, n_color, wscale=0.02):
        self.image_size = image_size
        self.n_color = n_color
        self.wscale = wscale
        self.conved_size = conved_image_size(image_size)
        super(Discriminator, self).__init__(
            c0=L.Convolution2D(
                **get_conv2d_kwargs(0, image_size, n_color, 512), wscale=wscale),
            c1=L.Convolution2D(
                **get_conv2d_kwargs(1, image_size, n_color, 512), wscale=wscale),
            c2=L.Convolution2D(
                **get_conv2d_kwargs(2, image_size, n_color, 512), wscale=wscale),
            c3=L.Convolution2D(
                **get_conv2d_kwargs(3, image_size, n_color, 512), wscale=wscale),
            l4=L.Linear(self.conved_size ** 2 * 512, 2, wscale=wscale),
            bn0=L.BatchNormalization(64),
            bn1=L.BatchNormalization(128),
            bn2=L.BatchNormalization(256),
            bn3=L.BatchNormalization(512),
        )

    def __call__(self, x, test=False):
        h = self.c0(x)
        h = F.leaky_relu(h)

        h = self.c1(h)
        h = self.bn1(h, test=test)
        h = F.leaky_relu(h)

        h = self.c2(h)
        h = self.bn2(h, test=test)
        h = F.leaky_relu(h)

        h = self.c3(h)
        h = self.bn3(h, test=test)
        h = F.leaky_relu(h)

        l = self.l4(h)

        return l

    def to_json(self):
        d = {
            'class_name': self.__class__.__name__,
            'kwargs': {
                'image_size': self.image_size,
                'n_color': self.n_color,
                'wscale': self.wscale,
            }
        }
        return json.dumps(d)
