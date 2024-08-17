from setuptools import setup, find_packages

setup(
    name='DiskVec',
    version='0.0.1',
    packages=find_packages(),
    install_requires=open('requirements.txt', 'r').read().splitlines(),
    python_requires='>=3.6',
)
