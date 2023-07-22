import os
from setuptools import setup, find_packages

requirementPath = 'requirements.txt'
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(
    name='diffusionModels',
    version='0.0.1',
    description='Theory of diffusion models',
    author='David Oliver Cortadellas',
    author_email='',
    url='',
    packages=find_packages(include=['utils']),
    install_requires=install_requires,
    )