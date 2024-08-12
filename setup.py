from setuptools import find_packages, setup

setup(
    name="diskhnsw",
    version="0.0.1",
    python_requires=">=3.9, <3.12", 
    license="APACHE 2.0",
    package_dir={'': 'diskhnsw'},
    packages=find_packages('diskhnsw'),
)