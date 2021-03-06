from setuptools import setup

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='satbenchmark',
    version='1.0.0',
    description='satbenchmark',
    packages=['satbenchmark'],
    install_requires=reqs.strip().split('\n'),
    include_package_data=True,
)