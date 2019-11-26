from .tabular import Tree as VAE_tree, CSV as VAE_csv
from .mnist import Mnist as VAE_mnist

__all__ = [VAE_csv, VAE_tree, VAE_mnist]