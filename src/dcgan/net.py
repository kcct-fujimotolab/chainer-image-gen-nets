import chainer
import chainer.functions as F
import chainer.links as L

n_z = 100


class Generator(chainer.Chain):

    def __init__(self, n_color, wscale=0.02):
        super(Generator, self).__init__(
            l0=L.Linear(n_z, 4 * 4 * 512, wscale=wscale),
            dc1=L.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=wscale),
            dc2=L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=wscale),
            dc3=L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=wscale),
            dc4=L.Deconvolution2D(
                64, n_color, 4, stride=2, pad=1, wscale=wscale),
            bn0l=L.BatchNormalization(4 * 4 * 512),
            bn0=L.BatchNormalization(512),
            bn1=L.BatchNormalization(256),
            bn2=L.BatchNormalization(128),
            bn3=L.BatchNormalization(64),
        )

    def __call__(self, z, test=False):
        h = self.l0(z)
        h = self.bn0l(h, test=test)
        h = F.relu(h)
        h = F.reshape(h, (z.data.shape[0], 512, 4, 4))

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


class Discriminator(chainer.Chain):

    def __init__(self, n_color, wscale=0.02):
        super(Discriminator, self).__init__(
            c0=L.Convolution2D(n_color, 64, 4, stride=2, pad=1, wscale=wscale),
            c1=L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=wscale),
            c2=L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=wscale),
            c3=L.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=wscale),
            l4=L.Linear(4 * 4 * 512, 2, wscale=wscale),
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
