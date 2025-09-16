from setuptools import setup, find_packages

setup(
    name='myflyteproject',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'flytekit',
        # other dependencies
    ],
)