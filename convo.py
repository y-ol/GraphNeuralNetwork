import tensorflow as tf
import numpy as np


def ref_A(edgelist):
    indices_A = list()
    for tuple in edgelist:
        indices_A.append(tuple[0])
    return tf.constant(indices_A, dtype=tf.int32)


def ref_B(edgelist):
    indices_B = list()
    for tuple in edgelist:
        indices_B.append(tuple[1])
    return tf.constant(indices_B, dtype=tf.int32)


def convolution(X, ref_A, ref_B):
    # undirected
    X_a = tf.gather(X, ref_A, axis=0)
    X_b = tf.gather(X, ref_B, axis=0)
    X_aggregate = tf.scatter_nd(tf.expand_dims(ref_A, axis=- 1), X_b, shape=tf.shape(X)) \
        + tf.scatter_nd(tf.expand_dims(ref_B, axis=-1),
                        X_a, shape=tf.shape(X)) + X

    return X_aggregate


def normalization(X, ref_A, ref_B):
    X_norm = tf.ones(shape=tf.shape(X)[0])
    d = convolution(X_norm, ref_A, ref_B) + 1
    d_rsqrt = tf.math.rsqrt(d)

    return ((tf.reshape(d_rsqrt, (-1, 1)) * X))  # -1 autocomplete
