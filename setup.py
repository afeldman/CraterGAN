#!/usr/bin/env python3

import os

# Third party
from setuptools import find_packages, setup
from apu.setup import setversion, Module

project_name="cratergan"

setversion(os.path.abspath(os.path.dirname(__file__)),
           f'{project_name}/__version__.py')

from cratergan import __author__, __email__
from cratergan.__version__ import VERSION

setup(
    name=project_name,
    version=VERSION,
    author=__author__,
    author_email=__email__,
    py_modules=[project_name],
    packages=find_packages(),
    include_package_data=True,
    project_urls={
        'Documentations':
        'https://github.com/afeldman/CraterGAN',
        'Source': 'https://github.com/afeldman/CraterGAN.git',
        'Tracker': 'https://github.com/afeldman/CraterGAN/issues'
    },
    install_requires=Module.load_requirements("requirements.txt"),
)
