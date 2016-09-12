import argparse
import glob
import os
import re

import chainer
import chainer.functions as F
import chainer.links as L
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from chainer import training
from chainer.training import extensions
from PIL import Image

import dcgan


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, required=True)
    parser.add_argument('--filename', type=str, default='{epoch}.png')
    args = parser.parse_args()

    n_column = 4
    n_row = 4
    n_img = n_column * n_row

    model_dir = 'result/model'
    output_dir = 'result/test'

    try:
        os.makedirs(output_dir)
    except:
        pass

    gen = dcgan.Generator()
    chainer.serializers.load_npz(
        '{}/{}/dcgan_model_gen.npz'.format(model_dir, args.epoch), gen)
    z = np.random.uniform(-1, 1, (n_img, dcgan.n_z)).astype(np.float32)

    y = gen(z, test=True)
    for i, img in enumerate(y.data):
        plt.subplot(n_row, n_column, i + 1)
        plt.imshow(((img + 1) / 2).transpose(1, 2, 0))
        plt.axis('off')
    filename = args.filename.format(epoch=args.epoch)
    plt.savefig('{}/{}'.format(output_dir, filename))
