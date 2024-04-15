phlash is a program for sampling from the posterior distribution of population size
history given whole genome sequence data. You can think of it as a Bayesian version
of Li & Durbin's popular [PSMC](https://github.com/lh3/psmc) program, with a more
modern interface and GPU acceleration built in.

## Requirements

- Python 3.10 or greater.

Optional but _highly recommended_:

- An NVIDIA GPU. Any relatively recent model should work. phlash has been tested on:
    - RTX 4090
    - A40
    - A100
    - V100

## Installation

phlash can be installed from PyPI using pip:

```
$ pip install phlash
```

If your machine contains an Nvidia GPU, you should install the `gpu` variant in order to enable it.
In this case, you should run:

```
$ pip install \
    -f "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html" \
    --extra-index-url "https://pypi.ngc.nvidia.com" \
    phlash[gpu]
```

I recommend installing phlash into a separate virtual environment: before running the
above, do

```
$ python3 -mvenv /path/to/phlash  # replace with desired path
$ source /path/to/phlash/bin/activate
$ pip3 install -U pip setuptools  # recent version of pip and setuptools are required
```
## Running the program
See [notebooks/example.md](notebooks/example.md) for example code and usage instructions.

## Troubleshooting / FAQ

I (Jonathan) am happy to assist you with using phlash, as much as my time allows.

- If you encounter a **bug** (program crash or other unexpected behavior) please
[file an issue](https://github.com/jthlab/phlash/issues/new) describing the bug.
- If you need help with anything else (installation, running the program, data
formatting, interpreting the output, etc.) please
[open a discussion](https://github.com/jthlab/phlash/discussions/new?category=q-a).

## Making Changes & Contributing

Contributions (in the form of Github pull requests) to improve the project are always welcome!

This project uses `pre-commit`_, please make sure to install it before making any
changes::

    pip install pre-commit
    cd phlash
    pre-commit install

It is a good idea to update the hooks to the latest version::

    pre-commit autoupdate
