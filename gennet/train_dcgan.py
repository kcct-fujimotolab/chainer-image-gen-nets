import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions

from gennet import dataset, util
from gennet.dcgan import generate, net
from gennet.dcgan.updater import DCGANUpdater


def main():
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
    parser.add_argument('--row', type=int, default=4)
    parser.add_argument('--col', type=int, default=4)
    parser.add_argument(
        '--slack-apikey', default=os.environ.get('SLACK_APIKEY'))
    parser.add_argument('--slack-channel',
                        default=os.environ.get('SLACK_CHANNEL'))
    args = parser.parse_args()

    if args.slack_apikey and args.slack_channel:
        print('# Post generated images to Slack (channel: {})'.format(
            args.slack_channel))
    else:
        print("# Don't post to Slack, but generated images will save into {}".format(
            args.out))

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

    util.save_model_json(gen, 'gen.model.json', output_dir=args.out)
    util.save_model_json(dis, 'dis.model.json', output_dir=args.out)

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
        filename='snapshot_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'dis/loss',
    ]))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.PlotReport(
        ['gen/loss', 'dis/loss'], 'epoch', file_name='loss.png'))
    if args.slack_apikey and args.slack_channel:
        trainer.extend(generate.generate_and_post_slack_extension(
            gen, args.row, args.col, args.out, args.slack_apikey, args.slack_channel), trigger=snapshot_interval)
    else:
        trainer.extend(generate.generate_image_extension(
            gen, args.row, args.col, args.out), trigger=snapshot_interval)

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
