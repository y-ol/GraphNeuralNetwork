import numpy as np
import tensorflow as tf
from tensorflow import keras
import spektral
from spektral.data import Dataset, Graph


dataset = spektral.datasets.qm9.QM9(amount=None, n_jobs=1)
