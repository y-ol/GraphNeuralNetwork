# Mean - Pooling
import tensorflow as tf
from tensorflow import keras
from abc import ABCMeta, abstractmethod
import convL
import convo


def add_graph_idx(input):  # segment ids
    n = input['num_nodes']
    N = tf.shape(n)[0]
    graph_idx = tf.repeat(tf.range(N), n)
    return {
        **input,
        "graph_idx": graph_idx,
        "graph_count": N}


class Pooling(keras.layers.Layer, metaclass=ABCMeta):
    def __init__(self, activation='relu'):
        super().__init__()
        self.activation = activation

    def call(self, input):
        input = add_graph_idx(input)
        X = input['X']
        graph_idx = input["graph_idx"]
        graph_count = input["graph_count"]
        return tf.math.unsorted_segment_mean(X, graph_idx, graph_count)
