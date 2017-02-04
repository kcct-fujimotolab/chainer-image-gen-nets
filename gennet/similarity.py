import argparse
import os

import chainer
import numpy as np

import gennet.dataset


class DatasetSimilarity(object):

    def __init__(self, a, b, device=None):
        self.xp = np if device < 0 else chainer.cuda.cupy
        self.a = self.xp.asarray(a)
        self.b = self.xp.asarray(b)

    def cosine(self):
        return self.xp.array(
            [self.xp.dot(a, b) / (self.xp.linalg.norm(a) * self.xp.linalg.norm(b))
             for a in self.a for b in self.b])

    def mean_squared_error(self):
        return self.xp.array([self.xp.average((a - b) ** 2) for a in self.a for b in self.b])


def split_dataset_arg(arg):
    n_colon = arg.count(':')
    num = None
    if n_colon == 0:
        dataset = arg
        filter_ = 'all'
    elif n_colon == 1:
        dataset, filter_ = arg.split(':')
    elif n_colon == 2:
        dataset, filter_, num = arg.split(':')
        num = int(num)
    else:
        raise RuntimeError("'{}' have too many ':'.".format(arg))

    return dataset, filter_, num


def load_dataset(dataset, filter_='all', shuffle=False):
    if os.path.exists(dataset):
        imgs = gennet.dataset.load(dataset, ndim=1)
        n_imgs = imgs.shape[0]
        split_index = int(n_imgs * 0.8)
        train, test = imgs[:split_index], imgs[split_index:]
    elif dataset == 'mnist':
        train, test = chainer.datasets.get_mnist(
            withlabel=False, ndim=1)
    elif dataset == 'cifar10':
        train, test = chainer.datasets.get_cifar10(
            withlabel=False, ndim=1)
    elif dataset == 'cifar100':
        train, test = chainer.datasets.get_cifar100(
            withlabel=False, ndim=1)
    else:
        raise RuntimeError(
            "'{}' is invalid value. The dataset should be exists path or 'mnist', 'cifar10' or 'cifar100'.".format(dataset))

    if filter_ == 'train':
        d = train
    elif filter_ == 'test':
        d = test
    elif filter_ == 'all':
        d = np.vstack((train, test))
    else:
        raise RuntimeError(
            "'{}' is invalid value. The filter should be 'train', 'test', or 'all'".format(filter_))

    if shuffle:
        np.random.shuffle(d)
    return d


def main():
    parser = argparse.ArgumentParser(
        description='Compute cosine similarity between 2 images')
    group = parser.add_argument_group('dataset',
                                      '''
        The npz formatted dataset file (*.npz), MNIST (mnist), CIFAR-10 (cifar10) or CIFAR-100 (cifar100).
        If you want to use only train or test data, then please add ":train" or ":test"
        after filepath (default is ":all").
        And you can add ":<number>" after it, this option will pick up <number> randomly from dataset.
        For example: 'dataset.npz:train', 'mnist:test', 'cifar10:all:50'
        ''')
    group.add_argument('a')
    group.add_argument('b')
    parser.add_argument(
        '--method', '-m', choices=('cosine', 'mse'), required=True)
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--head', type=int, default=50)
    args = parser.parse_args()

    xp = chainer.cuda.get_array_module(args.gpu)

    a_dataset, a_filter, a_num = split_dataset_arg(args.a)
    b_dataset, b_filter, b_num = split_dataset_arg(args.b)

    a = load_dataset(a_dataset, a_filter, shuffle=a_num is not None)[:a_num]
    b = load_dataset(b_dataset, b_filter, shuffle=b_num is not None)[:b_num]

    sim = DatasetSimilarity(a, b, device=args.gpu)

    if args.method == 'cosine':
        similarity = sim.cosine()
    elif args.method == 'mse':
        similarity = sim.mean_squared_error()

    similarity = chainer.cuda.to_cpu(similarity)
    similarity = np.sort(similarity)

    if args.method == 'cosine':
        similarity = similarity[::-1]

    for s in similarity[:args.head]:
        print(s)


if __name__ == '__main__':
    main()
