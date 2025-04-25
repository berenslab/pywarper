# pywarper

`pywarper` is a Python package for conformal mapping-based warping of neuronal arbor morphologies, based on the [MATLAB implementations](https://github.com/uygarsumbul/rgc) (Sümbül, et al. 2014). 

## Installation

To install the latest tagged version:

```bash
pip install pywarper
```

Or to install the development version, clone the repository and install it with `pip install -e`:

```bash
git clone https://github.com/berenslab/pywarper.git
pip install -e pywarper
```

By default, `pywarper` uses `scipy.sparse.linalg.spsolve` to solve sparse matrices, which can be slow. For better performance, you can manually install the additional dependencies of [scikit-sparse](https://github.com/scikit-sparse/scikit-sparse) first:

```bash
# mac
brew install suite-sparse

# debian
sudo apt-get install libsuitesparse-dev
```

then:

```bash
pip install pywarper[scikit-sparse]
```

## Usage

See [example notebooks](https://github.com/berenslab/pywarper/blob/main/notebooks/) for usage. 