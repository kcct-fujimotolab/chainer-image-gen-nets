import glob
import os
import re

import chainer
import chainer.functions as F
import chainer.links as L
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from chainer import training
from chainer.training import extensions
from PIL import Image

import dcgan

matplotlib.use('Agg')


if __name__ == '__main__':

    n_column = 4
    n_row = 4
    n_img = n_column * n_row

    model_dir = 'result/model'
    output_dir = 'result/test'

    try:
        os.makedirs(output_dir)
    except:
        pass

    model_epochs = set(ep for ep in os.listdir(model_dir))
    exists_epochs = set(re.search('(\d+).png', f).group(1)
                        for f in os.listdir(output_dir) if not f.startswith('.'))
    predict_epochs = model_epochs - exists_epochs

    print(predict_epochs)
    for epoch in predict_epochs:
        gen = dcgan.Generator()
        chainer.serializers.load_npz(
            '{}/{}/dcgan_model_gen.npz'.format(model_dir, epoch), gen)
        z = np.random.uniform(-1, 1, (n_img, dcgan.n_z)).astype(np.float32)

        y = gen(z, test=True)
        for i, img in enumerate(y.data):
            plt.subplot(n_row, n_column, i + 1)
            plt.imshow(((img + 1) / 2).transpose(1, 2, 0))
            plt.axis('off')
        plt.savefig('{}/{}.png'.format(output_dir, epoch))
