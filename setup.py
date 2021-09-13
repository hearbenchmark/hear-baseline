#!/usr/bin/env python3
from setuptools import find_packages, setup

long_description = open("README.md", "r").read()

setup(
    name="hearbaseline",
    description="Holistic Evaluation of Audio Representations (HEAR) 2021 -- Baseline Model",
    author="HEAR 2021 NeurIPS Competition Committee",
    author_email="deep@neuralaudio.ai",
    url="https://github.com/neuralaudio/hear-baseline",
    license="Apache-2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/neuralaudio/hear-baseline/issues",
        "Source Code": "https://github.com/neuralaudio/hear-baseline",
    },
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.7",
    install_requires=[
        "librosa",
        # otherwise librosa breaks
        "numba==0.48",
        # tf 2.6.0
        "numpy==1.19.2",
        "tensorflow>=2.0",
        "torch",
        # For wav2vec2 model
        "speechbrain",
        "transformers==4.4.0",
        "torchcrepe",
        "torchopenl3",
        # otherwise librosa breaks
        "numba==0.48",
        # "numba>=0.49.0", # not directly required, pinned by Snyk to avoid a vulnerability
        "scikit-learn>=0.24.2",  # not directly required, pinned by Snyk to avoid a vulnerability
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
            "pytest-env",
        ],
        "dev": [
            "pre-commit",
            "black",  # Used in pre-commit hooks
            "pytest",
            "pytest-cov",
            "pytest-env",
        ],
    },
)
