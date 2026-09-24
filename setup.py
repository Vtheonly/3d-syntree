"""Packaging metadata for the 3D-SynTree SBDD framework."""

from setuptools import setup, find_packages


import os

long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()





setup(
    name="syntree",
    version="0.1.0",
    description="Structure-Based Molecular Design via Reaction-Constrained Synthon Assembly",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="3D-SynTree Contributors",
    license="MIT",
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torch-geometric>=2.4.0",
        "rdkit>=2023.9.1",
        "pandas>=2.1.0",
        "pyarrow>=14.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "huggingface-hub>=0.20.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "test": ["pytest>=7.4.0"],
        "hdf5": ["h5py>=3.10.0"],
        "evaluation": ["aizynthfinder>=4.0.0"],
    },
    entry_points={
        "console_scripts": [
            "syntree=main:main",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)
