import os

import slacker

apikey = os.getenv('SLACK_APIKEY')
channel = os.getenv('SLACK_CHANNEL')
slack = slacker.Slacker(apikey)


def upload_img(imgfile):
    slack.files.upload(imgfile, channels=channel)
