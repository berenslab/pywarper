from importlib.metadata import PackageNotFoundError, version

from .warpers import Warper

__all__ = ["Warper"]

try:
    __version__ = version(__name__)       
except PackageNotFoundError:              
    __version__ = "0.0.0.dev0"
