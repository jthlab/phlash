## phlash

    phlash

## Requirmements
- Python 3.10 or greater.
- An NVIDIA GPU. Any relatively recent model should work. phlash has been tested on:
    - RTX 4090
    - A40
    - A100
    - V100

## Installation

The recommended way to install phlash is to into a separate virtual environment:

```
$ python3 -mvenv /path/to/phlash  # replace with desired path
$ source /path/to/phlash/bin/activate
$ pip3 install -U pip setuptools  # recent version of pip and setuptools are required
$ pip3 install git+https://github.com/jthlab/phlash@latest
```

phlash can also be installed from PyPI, however the usual `pip install phlash` will fail
because of dependence on CUDA and Jax. Instead, you must type:

```
$ pip install \
    -f "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html" \
    --extra-index-url "https://pypi.ngc.nvidia.com" \
    phlash
```

## Troubleshooting / FAQ

I (Jonathan) am happy to assist you with using phlash, as much as my time allows.

- If you encounter a **bug** (program crash or other unexpected behavior) please [file an issue](https://github.com/jthlab/phlash/issues/new) describing the bug.
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
