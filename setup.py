#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages


dependencies = [
    'pycuda',
    'matplotlib',
    'redis',
    'numpy',
    'fn',
    ]

packages = find_packages()


setup(
    name='desert',
    version='0.1.0',
    description='desert',
    url='https://github.com/inconvergent/desert',
    license='MIT',
    author='@inconvergent',
    author_email='inconvergent@gmail.com',
    install_requires=dependencies,
    packages=packages,
    package_data={
        'desert': [
            'cuda/agg.cu',
            'cuda/agg_bin.cu',
            'cuda/box.cu',
            'cuda/bzspl.cu',
            'cuda/circle.cu',
            'cuda/dot.cu',
            'cuda/stroke.cu',
            ],
    },
    entry_points={
        'console_scripts': [
            'erosion=erosion:run'
            ]
        },
    zip_safe=True,
    dependency_links=['git+https://github.com/inconvergent/fn.git@master#egg=fn-0'],
    )

