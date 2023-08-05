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


def dropout_mask(p, shape):
    p_list = (tf.random.uniform(
        shape=shape, minval=0, maxval=1, dtype=tf.dtypes.float32))
    mask = p_list >= p
    return tf.cast(mask, dtype=tf.dtypes.float32)


def get_shape(tensor, row_wise):
    """DropOut and NodeSampling"""
    if row_wise:
        shape = ((tf.shape(tensor)[0]), 1)
    else:
        shape = tf.shape(tensor)
    return shape


def get_edgedropshape(tensor, ref, row_wise):
    """DropEdge and GDC"""
    if row_wise:
        shape = (tf.shape(ref)[0], 1)
    else:
        shape = (tf.shape(ref)[0], tf.shape(tensor)[1])
    return shape


def convolution(X, ref_A,  ref_B, mask=None):
    """ Performs convolution with optional mask"""
    # X_a = tf.gather(X, ref_A, axis=0)
    X_b = tf.gather(X, ref_B, axis=0)
    if mask is not None:
        X_b *= mask
    X_aggregate = tf.scatter_nd(tf.expand_dims(
        ref_A, axis=- 1), X_b, shape=tf.shape(X))
    return X_aggregate


def normalization(X, ref_A, ref_B, mask=None, node_mask=False):

    if node_mask:
        X_norm = mask
        d = convolution(X_norm, ref_A, ref_B)
    else:
        X_norm = tf.ones(shape=(tf.shape(X)[0], tf.shape(
            mask)[1] if mask is not None else 1))  # num_nodes
        d = convolution(X_norm, ref_A, ref_B, mask)

    d_rsqrt = tf.math.rsqrt(d)
    d_rsqrt = tf.where(tf.math.is_inf(d_rsqrt), .0, (d_rsqrt))
    return d_rsqrt * X  # if mask && 1 dim value >1


def normalize_convo(X, ref_A, ref_B, mask=None, node_mask=False):
    """ Combines convolution and normalization in a single step with optional masking support"""
    conv_X = normalization(X, ref_A, ref_B, mask, node_mask)
    if node_mask:
        conv_X = convolution(conv_X * mask, ref_A, ref_B)
    else:
        conv_X = convolution(conv_X, ref_A, ref_B, mask)
    conv_X = normalization(conv_X, ref_A, ref_B, mask, node_mask)
    return X + conv_X  # self connection

# node mask --> true : NodeDropOut, false : EdgeDropOut/GDC
# mask = dropout_mask result
