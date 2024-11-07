import os

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',  'rome_dataset'))

# DATA_ROOT = "rome_dataset"


from .rome_dataset import *