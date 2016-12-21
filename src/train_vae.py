import argparse

import chainer
from chainer.training import extensions

import dataset
import vae.net


def main():
    parser = argparse.ArgumentParser(description='Training with VAE')
    parser.add_argument('dataset', help='The npz formatted dataset file')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    # parser.add_argument('--resume', '-r', default='',
    #                     help='Resume the training from snapshot')
    # parser.add_argument('--unit', '-u', type=int, default=1000,
    #                     help='Number of units')
    # parser.add_argument('--snapshot', type=int, nargs='*',
    #                     default=range(1, 10001, 10))
    # parser.add_argument('--filename', default='{epoch}.png')
    parser.add_argument('--dimz', '-z', type=int, default=20,
                        help='dimention of encoded vector')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('# dim z: {}'.format(args.dimz))
    print('')

    train = dataset.load(args.dataset, ndim=1)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    n_train, image_size = train.shape

    model = vae.net.VAE(image_size, args.dimz, 500)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    updater = chainer.training.StandardUpdater(
        train_iter, optimizer, loss_func=model.get_loss_func(), device=args.gpu)
    trainer = chainer.training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()


if __name__ == '__main__':
    main()
