# pywarper

`pywarper` is a Python package for conformal mapping-based warping of neuronal arbor morphologies, based on the [MATLAB implementations](https://github.com/uygarsumbul/rgc) (Sümbül, et al. 2014). 

## Installation

```bash
git clone https://github.com/berenslab/pywarper.git
pip install -e pywarper
```

By default, `pywarper` uses `scipy.sparse.linalg.spsolve` to solve sparse matrices, which can be slow. For faster performance, you can manually install [scikit-sparse](https://github.com/scikit-sparse/scikit-sparse), as it requires additional dependencies:

```bash
# mac
brew install suite-sparse

# debian
sudo apt-get install libsuitesparse-dev

pip install -e pywarper[scikit-sparse]
```

## Usage

See the [example](https://github.com/berenslab/pywarper/blob/main/notebooks/example.ipynb) notebook for usage. 