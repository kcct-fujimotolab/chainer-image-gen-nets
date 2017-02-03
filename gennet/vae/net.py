import json
import math

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import six
from chainer.functions.loss.vae import gaussian_kl_divergence
from PIL import Image


class VAE(chainer.Chain):

    def __init__(self, n_in, n_latent, n_h, n_color=3, C=1.0, k=1):
        """
        Args:
            n_in (int): The number of input layers
            n_latent (int): The number of latent layers (encoded layers)
            n_h (int): The number of hidden layers
            n_color (int): The number of color of training images
            C (float): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """
        self.n_in = n_in
        self.n_latent = n_latent
        self.n_h = n_h
        self.n_color = n_color
        self.C = C
        self.k = k
        super(VAE, self).__init__(
            # encoder
            le1=L.Linear(n_in, n_h),
            le2_mu=L.Linear(n_h, n_latent),
            le2_ln_var=L.Linear(n_h, n_latent),
            # decoder
            ld1=L.Linear(n_latent, n_h),
            ld2=L.Linear(n_h, n_in),
        )

    def __call__(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = F.tanh(self.le1(x))
        mu = self.le2_mu(h1)
        ln_var = self.le2_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h1 = F.tanh(self.ld1(z))
        h2 = self.ld2(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def loss_func(self, x, train=True):
        """The loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        """
        mu, ln_var = self.encode(x)
        batchsize = len(mu.data)

        reconstruction_loss = 0
        for l in six.moves.range(self.k):
            z = F.gaussian(mu, ln_var)
            reconstruction_loss += F.bernoulli_nll(
                x, self.decode(z, sigmoid=False)) / (self.k * batchsize)

        loss = reconstruction_loss + \
            self.C * gaussian_kl_divergence(mu, ln_var) / batchsize
        chainer.report({'loss': loss}, self)
        return loss

    def make_hidden(self, batchsize):
        return np.random.uniform(-1, 1, (batchsize, self.n_latent)).astype(np.float32)

    def _ndarray_to_image(self, x):
        x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
        rows = cols = int(math.sqrt(x.shape[0]))
        W = H = int(math.sqrt(x.shape[1] / self.n_color))
        x = x.reshape((rows, cols, self.n_color, H, W))
        x = x.transpose(0, 3, 1, 4, 2)
        if self.n_color == 1:  # grayscale
            x = x.reshape((rows * H, cols * W))
        else:
            x = x.reshape((rows * H, cols * W, self.n_color))
        return Image.fromarray(x)

    def make_images(self, x, rows=4, cols=4):
        np.random.seed(0)
        n_images = rows * cols

        x = np.random.permutation(x)[:n_images]
        x = chainer.Variable(self.xp.asarray(x))
        x_reconstruct = self.decode(self.encode(x)[0])
        x = chainer.cuda.to_cpu(x.data)
        x_reconstruct = chainer.cuda.to_cpu(x_reconstruct.data)
        np.random.seed()

        return self._ndarray_to_image(x), self._ndarray_to_image(x_reconstruct)

    def make_random_images(self, rows=4, cols=4):
        np.random.seed(0)
        n_images = rows * cols
        z = chainer.Variable(self.xp.asarray(self.make_hidden(n_images)))
        x = self.decode(z)
        x = chainer.cuda.to_cpu(x.data)
        np.random.seed()

        return self._ndarray_to_image(x)

    def to_json(self):
        d = {
            'class_name': self.__class__.__name__,
            'kwargs': {
                'n_in': self.n_in,
                'n_h': self.n_h,
                'n_latent': self.n_latent,
                'n_color': self.n_color,
                'C': self.C,
                'k': self.k,
            }
        }
        return json.dumps(d)
