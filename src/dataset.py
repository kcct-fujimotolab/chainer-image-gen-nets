import argparse
import glob
import os

import numpy
import PIL.Image


def convert(files, size=(64, 64)):
    include_exts = ['.jpg', '.jpeg', '.gif', '.png']
    files = [f for f in files if os.path.splitext(f)[1] in include_exts]
    images = numpy.empty(
        (len(files), 3, size[0], size[1]), dtype=numpy.float32)

    for i, f in enumerate(files):
        img = PIL.Image.open(f).convert('RGB').resize(size)
        img = numpy.asarray(img).astype(numpy.float32).transpose(2, 0, 1)
        images[i] = img

    return images


def load(filename, ndim=2):
    images = numpy.load(filename)['img']
    if ndim == 1:
        images = images.reshape(-1,
                                images.shape[1], images.shape[2] * images.shape[3])
    return images


def save(filename, images):
    numpy.savez(filename, img=images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('--output', '-o', default='dataset.npz')
    args = parser.parse_args()

    files = glob.iglob(os.path.join(args.input_dir, '*'))
    images = convert(files)
    save(args.output, images)


if __name__ == '__main__':
    main()
