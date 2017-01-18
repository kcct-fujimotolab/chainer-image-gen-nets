import slacker


def upload_img(apikey, channel, imgfile):
    slack = slacker.Slacker(apikey)
    slack.files.upload(imgfile, channels=channel)
