#!/usr/bin/env python
from setuptools import setup,find_packages

def parse_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

setup(
    name="BiosynthPipeline",

    version="1.0",

    description="A combined retrobiosynthesis pipeline that seamlessly integrates Retrotide and Pickaxe."
                "Retrotide is a synthesis planning software for designing chimeric type I polyketide synthases (PKSs)"
                "Meanwhile, Pickaxe is a synthesis planning software catered specifically for enzymatic synthesis",

    author="Yash Chainani and Tyler Backman",

    author_email="ychainani@lbl.gov",

    url="https://github.com/JBEI/DemoRemo",

    packages = find_packages(),

    install_requires=parse_requirements('requirements.txt'), # + ['map4 @ git+https://github.com/reymond-group/map4@v1.0'],

    package_dir={},

    package_data={'retrobiosynthesisPipeline':['data/coreactants_and_rules/all_cofactors_updated.csv',
                                               'data/coreactants_and_rules/all_cofactors.tsv'
                                               'data/coreactants_and_rules/JN1224MIN_rules.tsv',
                                               'data/coreactants_and_rules/JN3604IMT_rules.tsv',
                                               'data/all_known_metabolites.txt',
                                               'models/xgboost_ecfp4_2048_4_add_concat.pkl',
                                               'models/xgboost_ecfp4_2048_4_add_subtract.pkl',
                                               'models/xgboost_ecfp4_2048_4_by_ascending_MW.pkl',
                                               'models/xgboost_ecfp4_2048_4_by_descending_MW.pkl']},

    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research/Development",
        "Intended Audience :: Scientific Engineering",
        "Intended Audience :: Application",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
    ],
)
