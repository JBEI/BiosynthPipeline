#!/usr/bin/env python

from setuptools import setup

setup(
    name='retrotide',
    version='0.1',
    description='Biosynthetic Cluster Simulator',
    author='The Quantitative Metabolic Modeling group',
    author_email='tbackman@lbl.gov',
    url='https://github.com/JBEI/BiosyntheticClusterSimulator',
    packages=['retrotide'],
    install_requires=[
	'cobra', 
	'numpy >= 1.8.0',
	],
    package_data={'retrotide': ['data/*']},
    license='see license.txt file',
    keywords = ['biochemistry', 'synthetic biology'],
    classifiers = [],
    python_requires='>=3.6',
    )
