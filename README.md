![HEAR2021](https://neuralaudio.ai/assets/img/hear-header-sponsor.jpg)
# HEAR 2021 Starter Kit

### NeurIPS Competition
The HEAR 2021 challenge invites you to create an audio embedding that is as holistic as
the human ear, i.e., one that performs well across a variety of everyday domains.
The challenge starts with three diverse and approachable open tasks, but also includes
a variety of held-out secret tasks. The three open tasks are: word classification,
pitch detection, and sound event detection. Each is relatively simple on its own.
Our twist is asking you to solve them all at once.

Teams will develop an embedding of arbitrary size to be fed into a generic predictor
by our evaluation algorithm. This predictor will be shallowly trained for each team
and each task.

For full competition details please visit the
[competition website.](https://neuralaudio.ai/hear2021-holistic-evaluation-of-audio-representations.html)

**Submissions are open!**
[Submit your entry](https://docs.google.com/forms/d/e/1FAIpQLSfSz7l4Aohg4JD_TTqKcIOkejM_ws0ho4kfD2nDeKQ4YWz5RA/viewform?usp=sf_link)
prior to July 15th 2021 AoE to be included in the first leaderboard update.
We will be holding monthly leaderboard updates up until the final submission
deadline of October 15th 2021.

## Starter Kit
This repository contains code to help participants start building and testing a python3
module and PyTorch / TensorFlow audio embedding model for the HEAR 2021 competition.
This repo is also setup as a pip3 installable package to demonstrate the submission
requirements. Contents of the starter-kit include:
- Baseline audio embedding model
- Script to validate modules against the competition API requirements
- Example setup.py file to demonstrate packaging for submission

### Installation
This starter-kit can be installed using a
local source tree pip install, which is the same method that will be used by the
organizers to install competition submissions.
You can install the starter kit as follows:
1) `git clone https://github.com/neuralaudio/hear-starter-kit.git`
2) `python3 -m pip install ./hear-starter-kit`

This will install a package called `hearkit` and all the required dependencies.

### Baseline
The baseline model is located at [hearkit/baseline.py](hearkit/baseline.py). It is a
DSP-based implementation that uses a random projection on mel-spectrums to create
audio embeddings with a dimensionality of 4096. The baseline model implements the
[common API](https://neuralaudio.ai/hear2021-holistic-evaluation-of-audio-representations.html#common-api)
that is required for competition submissions.

### Validation Script
The validation script checks a module against the
[common API](https://neuralaudio.ai/hear2021-holistic-evaluation-of-audio-representations.html#common-api).
It accepts any module as input and verifies that the correct functions are exposed and
produce an output that is formed correctly. To run the validation
script against a module:
```
python3 -m hearkit.validate <module-to-test> -m <path-to-model-checkpoint-file>
```
Example usage with the baseline, which contains no model checkpoint weights:
```python
python3 -m hearkit.validate hearkit.baseline
```

### Packaging
A goal of HEAR 2021 is to promote the development of general purpose audio embeddings
that are easy to access and easy to use. As such, we require that all submissions are
pip installable packages. This starter-kit provides an example on how to create a
minimal pip installable package. The important components of this are:
1) The `hearkit` package, which is the `hearkit` subfolder containing a `__init__.py`
    file.
2) The `setup.py` python file which is the main config file for installation.

To create your own package for submission you can fork this repo, rename the `hearkit`
subfolder to your package name, and update the fields in the `setup.py` config to the
reflect your package and team.

You can then install your package locally using:
```python
python3 -m pip install <path-to-project-root>
```
Where the project root is the folder containing `setup.py`.

If your current working directory is the root of your project you can run:
```python
python3 -m pip install .
```

To install in developer mode while you are working on your model:
```python
python3 -m pip install -e <path-to-project-root>
```

For a more detailed tutorial on packaging we recommend checking out this
[python-packaging tutorial](https://python-packaging.readthedocs.io/en/latest/index.html).

We realize that the pip installable requirement may pose a challenge to some entrants.
If this criterion poses an issue for you, the HEAR team would be glad to help. Please
each out to us by email at deep-at-neuralaudio.ai.

## Development
Clone repo:
```
git clone https://github.com/neuralaudio/hear-starter-kit.git
cd hear-start-kit
```
Install in development mode:
```
python3 -m pip install -e ".[dev]"
```
Install pre-commit hooks:
```
pre-commit install
```
