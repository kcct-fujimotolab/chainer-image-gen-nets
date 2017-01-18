import os

import chainer
import chainer.cuda
import numpy as np
from chainer import Variable
from PIL import Image


def make_image(gen, dis, rows, cols, output_dir):
    np.random.seed(0)
    n_images = rows * cols
    xp = gen.xp
    z = Variable(xp.asarray(gen.make_hidden(n_images)))
    x = gen(z, test=True)
    x = chainer.cuda.to_cpu(x.data)
    np.random.seed()

    x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
    _, _, H, W = x.shape
    x = x.reshape((rows, cols, 3, H, W))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((rows * H, cols * W, 3))

    preview_dir = os.path.join(output_dir, 'preview')
    preview_path = os.path.join(
        preview_dir, 'image{:0>8}.png'.format(trainer.updater.epoch))
    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)
    Image.fromarray(x).save(preview_path)
    return preview_path


def generate_image_extension(gen, dis, rows, cols, output_dir):
    @chainer.training.make_extension()
    def mkimg(trainer):
        make_image(gen, dis, rows, cols, output_dir)

    return mkimg


def generate_and_post_slack_extension(gen, dis, rows, cols, output_dir, apikey, channel):
    @chainer.training.make_extension()
    def post(trainer):
        img_path = make_image(gen, dis, rows, cols, output_dir)
        upload_img(apikey, channel, img_path)

    return post
