#!/usr/bin/env python3
from setuptools import find_packages, setup

long_description = open("README.md", "r", encoding="utf-8").read()

setup(
    name="hearbaseline",
    version="2021.0.3",
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
        "tensorflow",
        "torch",
        # For wav2vec2 model
        "speechbrain",
        "transformers==4.4.0",
        #"ruamel.base",
        #"ruamel.std.argparse",
        "absl_py==0.13.0",
        "astunparse==1.6.3",
        "click==8.0.1",
        "flatbuffers==1.12",
        "gast==0.4.0",
        "google-pasta==0.2.0",
        "grpcio==1.39.0",
        "huggingface_hub==0.0.15",
        "hyperpyyaml==1.0.0",
        "pyyaml==5.4.1",
        "ruamel.yaml.clib==0.2.6",
        "ruamel.yaml==0.17.14",
        "sentencepiece==0.1.96",
        "sacremoses==0.0.45",
        "urllib3==1.26.6",
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
