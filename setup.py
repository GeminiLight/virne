#!/usr/bin/env python3

# pylint: disable=missing-module-docstring

import pathlib
import re
import sys

import setuptools
from setuptools import setup


VERSION_CONTENT = None

try:
    setup(
        name='virne',
        version='0.4.0',
        author='GeminiLight',
        author_email='wtfly2018@gmail.com',
        description='Virne is a unified framework for virtual network embedding.',
        url='https://github.com/GeminiLight/virne',
        packages=setuptools.find_namespace_packages(
            include=['', 'virne', 'virne.*'],
        ),
        include_package_data=True,
    )
finally:
    raise ValueError