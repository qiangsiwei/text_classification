import os
from setuptools import setup

base_dir = os.path.dirname(os.path.abspath(__file__))

setup(name = 'text_classification',
    version = '0.1',
    description = 'keras implementation of text classification algorithms',
    author = 'Qiang Siwei',
    author_email = 'qiangsiwei@outlook.com',
    url = '',
    packages = ['text_classification',\
                'text_classification.layers',\
                'text_classification.models'],
    long_description = open(os.path.join(base_dir,'README.md')).read(),
    test_suite = 'tests.get_tests',
) 