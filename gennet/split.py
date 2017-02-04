import argparse
import os

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--size', '-s', nargs=2, type=int, required=True)
    parser.add_argument('--out', '-o', default='out')
    args = parser.parse_args()

    img = Image.open(args.input)
    img = np.asarray(img)
    H, W, n_color = img.shape
    row = int(W / args.size[0])
    col = int(H / args.size[1])

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    for x in range(row):
        for y in range(col):
            i = x * row + y
            _x = x * args.size[0]
            _y = y * args.size[1]
            _img = img[_x:_x + args.size[0], _y:_y + args.size[1]]
            filepath = os.path.join(args.out, '{}.png'.format(i))
            Image.fromarray(_img).save(filepath)


if __name__ == '__main__':
    main()
