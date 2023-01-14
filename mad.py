import numpy as np


def vector_length(array1):
    return np.sqrt(np.sum(array1 ** 2, axis=-1))


def matrix_product(X):
    return np.matmul(X, X.T)


def cosine(X):
    return 1 - matrix_product(X/vector_length(X).reshape((-1, 1)))


def ad(X):
    return np.mean(cosine(X))
