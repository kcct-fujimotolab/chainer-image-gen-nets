import json
import os

from gennet import dcgan, vae


def model_from_json_file(json_file):
    with open(json_file, 'r') as f:
        j = f.read()
    return model_from_json(j)


def model_from_json(json_string):
    d = json.loads(json_string)
    class_name = d.get('class_name')
    kwargs = d.get('kwargs')

    if class_name == 'VAE':
        layer_class = vae.net.VAE
    elif class_name == 'Generator':
        layer_class = dcgan.net.Generator
    elif class_name == 'Discriminator':
        layer_class = dcgan.net.Discriminator
    else:
        raise AttributeError("Unknown class_name: '{}'".format(class_name))

    return layer_class(**kwargs)


def save_model_json(model, filename, output_dir='.'):
    if not hasattr(model, 'to_json'):
        raise NotImplementedError(
            "'{}' has no to_json() method".format(type(model).__name__))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, filename), 'w') as f:
        d = model.to_json()
        f.write(d)
