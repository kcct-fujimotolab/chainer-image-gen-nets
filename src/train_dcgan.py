import argparse

import chainer
import numpy
from chainer import training
from chainer.training import extensions

import dataset
from dcgan import net
from dcgan.updater import DCGANUpdater


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trainning with DCGAN')
    dataset_options = parser.add_mutually_exclusive_group(required=True)
    dataset_options.add_argument(
        '--dataset', '-d', help='The npz formatted dataset file')
    dataset_options.add_argument(
        '--use-mnist', action='store_true', help='Use mnist dataset')
    dataset_options.add_argument(
        '--use-cifar10', action='store_true', help='Use CIFAR-10 dataset')
    dataset_options.add_argument(
        '--use-cifar100', action='store_true', help='Use CIFAR-100 dataset')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--snapshot_interval', type=int, default=50)
    parser.add_argument('--filename', default='{epoch}.png')
    args = parser.parse_args()

    if args.dataset:
        train = dataset.load(args.dataset, ndim=3)
    elif args.use_mnist:
        train, _ = chainer.datasets.get_mnist(
            withlabel=False, scale=255., ndim=3)
    elif args.use_cifar10:
        train, _ = chainer.datasets.get_cifar10(
            withlabel=False, scale=255., ndim=3)
    elif args.use_cifar100:
        train, _ = chainer.datasets.get_cifar100(
            withlabel=False, scale=255., ndim=3)

    n_train, n_color, width, height = train.shape

    gen = net.Generator(width, n_color)
    dis = net.Discriminator(width, n_color)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()

    optimizer_gen = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
    optimizer_dis = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
    optimizer_gen.setup(gen)
    optimizer_dis.setup(dis)
    optimizer_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))
    optimizer_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    updater = DCGANUpdater(gen, dis, iterator=train_iter, optimizer={
                           'gen': optimizer_gen, 'dis': optimizer_dis}, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'epoch')
    trainer.extend(extensions.snapshot(
        filename='snapshot_iter_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'dis/loss',
    ]))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()
