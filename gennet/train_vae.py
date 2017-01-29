import argparse
import os

import chainer
from chainer.training import extensions

from gennet import dataset, util
from gennet.vae import net


def main():
    parser = argparse.ArgumentParser(description='Training with VAE')
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
    parser.add_argument('--snapshot_interval', '-s', type=int, default=1000,
                        help='Interval of snapshot')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('# dim z: {}'.format(args.dimz))
    print('')

    if args.dataset:
        imgs = dataset.load(args.dataset, ndim=3)
        n_imgs = imgs.shape[0]
        split_index = int(n_imgs * 0.8)
        train, test = imgs[:split_index], imgs[split_index:]
    elif args.use_mnist:
        train, test = chainer.datasets.get_mnist(
            withlabel=False, ndim=3)
    elif args.use_cifar10:
        train, test = chainer.datasets.get_cifar10(
            withlabel=False, ndim=3)
    elif args.use_cifar100:
        train, test = chainer.datasets.get_cifar100(
            withlabel=False, ndim=3)

    n_train, n_color, H, W = train.shape
    train = train.reshape(-1, n_color * H * W)
    test = test.reshape(-1, n_color * H * W)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    model = net.VAE(n_color * W * H, args.dimz, 500, n_color=n_color)
    util.save_model_json(model, 'vae.model.json', output_dir=args.out)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    updater = chainer.training.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = chainer.training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'epoch')

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss']))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))

    @chainer.training.make_extension()
    def save_images(trainer):
        out_dir = os.path.join(
            trainer.out, 'preview_epoch_{}'.format(trainer.updater.epoch))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        img, img_r = model.make_images(train)
        img.save(os.path.join(out_dir, 'train.png'))
        img_r.save(os.path.join(out_dir, 'train_reconst.png'))

        img, img_r = model.make_images(test)
        img.save(os.path.join(out_dir, 'test.png'))
        img_r.save(os.path.join(out_dir, 'test_reconst.png'))

        model.make_random_images().save(os.path.join(out_dir, 'random.png'))

    trainer.extend(save_images, trigger=snapshot_interval)

    trainer.run()


if __name__ == '__main__':
    main()
