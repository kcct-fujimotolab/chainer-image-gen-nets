from setuptools import setup

setup(name='chainer-image-gen-nets',
      version='0.0.1',
      description='Implementation of Image Generative Networks with Chainer',
      url='https://github.com/kcct-fujimotolab/chainer-image-gen-nets',
      author='Hayato Kawai',
      author_email='fohte.hk@gmail.com',
      license='MIT License',
      packages=[
          'gennet',
      ],
      install_requires=[
          'chainer',
          'Pillow',
          'numpy',
          'slacker',
      ],
      )
