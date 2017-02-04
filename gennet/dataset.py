import argparse
import glob
import os

import numpy
import PIL.Image


def convert(files):
    include_exts = ['.jpg', '.jpeg', '.gif', '.png']
    files = [f for f in files if os.path.splitext(f)[1] in include_exts]

    images = numpy.asarray([
        numpy.asarray(PIL.Image.open(f).convert('RGB')).astype(
            numpy.float32).transpose(2, 0, 1)
        for f in files])

    return images


def load(filename, ndim=3, scale=1.):
    images = numpy.load(filename)['img']
    images *= scale / 255
    n_color = images.shape[1]
    width, height = images.shape[2:]
    if ndim == 1:
        return images.reshape(-1, n_color * width * height)
    if ndim == 2:
        return images.reshape(-1, n_color, width * height)
    if ndim == 3:
        return images
    return images


def save(filename, images, compress=True):
    if compress:
        numpy.savez_compressed(filename, img=images)
    else:
        numpy.savez(filename, img=images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('--output', '-o', default='dataset.npz')
    parser.add_argument('--compress', '-c', action='store_true')
    args = parser.parse_args()

    files = glob.iglob(os.path.join(args.input_dir, '*'))
    images = convert(files)
    save(args.output, images, compress=args.compress)
    print('{} files compressed and saved as {}. (shape: {})'.format(
        len(images), args.output, images.shape))


if __name__ == '__main__':
    main()
