import os
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), "README.md")) as fh:
    long_description = fh.read()

setup(
    name='minkf',
    version='0.0.3',
    packages=['minkf'],
    url='https://github.com/solbes/minkf',
    download_url='https://github.com/solbes/minkf/archive/refs/tags/0.0.3.zip',
    license='MIT',
    author='Antti Solonen',
    author_email='antti.solonen@gmail.com',
    description='Kalman filter, nothing more',
    keywords=['kalman filter', 'state estimation', 'kalman smoother'],
    install_requires=['numpy'],
    extras_require={
        'dev': ['pytest']
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "License :: OSI Approved :: MIT License",
        'Programming Language :: Python :: 3',
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
