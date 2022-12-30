from setuptools import find_packages, setup

setup(
    name='event_touch',
    packages=find_packages(),
    version='0.1.0',
    description='Event camera algorithms for optical touch sensing',
    author='Ronan Lawlor',
    license='MIT',
    setup_requires=['setuptools_scm'],
    include_package_data=True,
)