from setuptools import setup, find_packages

setup(
    name='TResnet',
    version='0.1',
    description='ResnetX with transformer applied at the output feature map of L-1',
    author='Ferenc Lippai',
    author_email='ferenc.lippai@gmail.com',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='githublink',
    install_requires=['pytorch-lightning', 'torch', 'torchvision',],
    packages=find_packages(),
)

