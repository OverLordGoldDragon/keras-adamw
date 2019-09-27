import os
import re
import codecs
from setuptools import setup, find_packages

current_path = os.path.abspath(os.path.dirname(__file__))


def read_file(*parts):
    with codecs.open(os.path.join(current_path, *parts), 'r', 'utf8') as reader:
        return reader.read()


def get_requirements(*parts):
    with codecs.open(os.path.join(current_path, *parts), 'r', 'utf8') as reader:
        return list(map(lambda x: x.strip(), reader.readlines()))


def find_version(*file_paths):
    version_file = read_file(*file_paths)
    version_matched = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_matched:
        return version_matched.group(1)
    raise RuntimeError('Unable to find version')


setup(
    name='keras-adamw',
    version=find_version('keras_adamw', '__init__.py'),
    packages=find_packages(),
    url='https://github.com/OverLordGoldDragon/keras-adamw',
    license='MIT',
    author='OverLordGoldDragon',
    description='Keras implementation of AdamW, SGDW, NadamW, Warm Restarts, and Learning Rate multipliers',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    install_requires=get_requirements('requirements.txt'),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
)
