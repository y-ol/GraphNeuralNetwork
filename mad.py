import numpy as np
from tensorflow.python.ops.numpy_ops import np_config

# Paper: Special case of M_{tgt}, where all values are 1 (nothing masked--> global)
# --> Description of my case sufficient
# eq 2 not here
# D_{tgt} = D, bc no mask
# 3,4 normalization merged in code
np_config.enable_numpy_behavior()


def vector_length(array1):
    return np.sqrt(np.sum(array1 ** 2, axis=-1))


def matrix_product(X):
    return np.matmul(X, X.T)


def cosine(X):
    return 1 - matrix_product(X/vector_length(X).reshape((-1, 1)))


def ad(X):
    return np.mean(cosine(X))
