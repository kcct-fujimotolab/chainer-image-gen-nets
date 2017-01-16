import argparse
import os

import chainer
import chainer.functions as F
import numpy
import progressbar

import dataset
import post_slack
from dcgan import generate, net


def random_indexes(n):
    xs = numpy.arange(n)
    numpy.random.shuffle(xs)
    return xs


if __name__ == '__main__':

    batchsize = 100
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
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--snapshot', type=int, nargs='*',
                        default=range(1, 10001, 10))
    parser.add_argument('--filename', default='{epoch}.png')
    args = parser.parse_args()

    xp = chainer.cuda.cupy if args.gpu >= 0 else numpy

    model_dir = '{}/model'.format(args.out)

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

    gen = net.Generator(n_color)
    dis = net.Discriminator(n_color)

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

    sum_loss_dis = xp.float32(0)
    sum_loss_gen = xp.float32(0)

    progress = progressbar.ProgressBar()
    for epoch in progress(range(args.epoch)):
        perm = random_indexes(n_train)
        for i in range(0, n_train - (n_train % batchsize), batchsize):
            z = chainer.Variable(
                xp.random.uniform(-1, 1, (batchsize, net.n_z)).astype(xp.float32))
            y_gen = gen(z)
            y_dis = dis(y_gen)
            loss_gen = F.softmax_cross_entropy(
                y_dis, chainer.Variable(xp.zeros(batchsize, dtype=xp.int32)))
            loss_dis = F.softmax_cross_entropy(
                y_dis, chainer.Variable(xp.ones(batchsize, dtype=xp.int32)))

            images = train[perm[i:i + batchsize]]
            if args.gpu >= 0:
                images = chainer.cuda.to_gpu(images)
            y_dis = dis(chainer.Variable(images))
            loss_dis += F.softmax_cross_entropy(
                y_dis, chainer.Variable(xp.zeros(batchsize, dtype=xp.int32)))

            optimizer_gen.zero_grads()
            loss_gen.backward()
            optimizer_gen.update()

            optimizer_dis.zero_grads()
            loss_dis.backward()
            optimizer_dis.update()

            sum_loss_gen += loss_gen.data.get()
            sum_loss_dis += loss_dis.data.get()

        if epoch + 1 in args.snapshot:
            outdir = '{}/{}'.format(model_dir, epoch + 1)
            try:
                os.makedirs(outdir)
            except:
                pass
            chainer.serializers.save_npz(
                '{}/dcgan_model_gen.npz'.format(outdir), gen)
            chainer.serializers.save_npz(
                '{}/dcgan_model_dis.npz'.format(outdir), dis)
            chainer.serializers.save_npz(
                '{}/dcgan_optimizer_gen.npz'.format(outdir), optimizer_gen)
            chainer.serializers.save_npz(
                '{}/dcgan_optimizer_dis.npz'.format(outdir), optimizer_dis)

            filename = args.filename.format(epoch=(epoch + 1))
            generate.generate(epoch + 1, filename=filename)
            post_slack.upload_img(
                '{}/test/{}'.format(args.out, filename))
