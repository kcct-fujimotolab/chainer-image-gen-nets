# chainer-image-gen-nets
An Implementation of Image Generative Networks with Chainer.

## Requirements
- Python3

## Setup
You can send auto-generated images with trained models to the Slack channel when training models.
If then, please set environment variables:
```sh
export SLACK_APIKEY=<slackbot token>
export SLACK_CHANNEL=<posting channel>
```

## Usage

### Train DCGAN model
Training model with [Deep Convolutional Generative Adversarial Network](https://arxiv.org/abs/1511.06434):
```
python src/train_dcgan.py dataset -e 3000 -g 0 --snapshot 1 5 10 50 100 500 1000 2000 3000 --filename 'dcgan_{epoch}.png'
```
