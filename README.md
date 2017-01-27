# chainer-image-gen-nets
An Implementation of Image Generative Networks with Chainer.

## Requirements
- Python3

## Installation
```sh
git clone https://github.com/kcct-fujimotolab/chainer-image-gen-nets.git
cd chainer-image-gen-nets
python setup.py develop
```

## Setup
You can send auto-generated images with trained models to the Slack channel when training models.
If then, please set environment variables:
```sh
export SLACK_APIKEY=<slackbot token>
export SLACK_CHANNEL=<posting channel>
```

## Usage

### Train VAE model
Training model with [Variational Auto-Encoder](https://arxiv.org/abs/1312.6114):
```
python gennet/train_vae.py --use-mnist -e 300 -g 0 -o out/vae/mnist --snapshot_interval 20
```

### Train DCGAN model
Training model with [Deep Convolutional Generative Adversarial Network](https://arxiv.org/abs/1511.06434):
```
python gennet/train_dcgan.py --use-mnist -e 300 -g 0 -o out/dcgan/mnist --snapshot_interval 20 --slack-channel @fohte
```
