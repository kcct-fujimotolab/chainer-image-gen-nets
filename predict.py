import argparse
import os

import chainer
import matplotlib; matplotlib.use('Agg') # isort:skip
import matplotlib.pyplot as plt # isort:skip
import numpy as np

import dcgan


def predict(epoch, filename='{epoch}.png'):
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
        '{}/{}/dcgan_model_gen.npz'.format(model_dir, epoch), gen)
    z = np.random.uniform(-1, 1, (n_img, dcgan.n_z)).astype(np.float32)

    y = gen(z, test=True)
    for i, img in enumerate(y.data):
        plt.subplot(n_row, n_column, i + 1)
        plt.imshow(((img + 1) / 2).transpose(1, 2, 0))
        plt.axis('off')
    filename = filename.format(epoch=epoch)
    plt.savefig('{}/{}'.format(output_dir, filename))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, required=True)
    parser.add_argument('--filename', type=str, default='{epoch}.png')
    args = parser.parse_args()

    predict(args.epoch, args.filename)
