import os

from setuptools import find_packages, setup

__version__ = "0.0.1"

setup(
    name="raccoon_gym",
    description="Set of robotic environments used in Workspace Raccoon",
    author="Chia-Ching Yen",
    author_email="ccyen@umich.edu",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    include_package_data=True,
    version=__version__,
    install_requires=["gymnasium>=0.26", "pybullet", "numpy", "scipy"],
    extras_require={
        "develop": ["pytest-cov", "black", "isort", "pytype", "sphinx", "sphinx-rtd-theme"],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
