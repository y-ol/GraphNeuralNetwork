# Mean - Pooling
import tensorflow as tf
from tensorflow import keras
from abc import ABCMeta, abstractmethod
import convL
import convo
"""
Pooling takes as input output from convo
  return {
            **inputs,  # decomposition
            'X': tf.nn.bias_add(c.convolution(X, ref_A, ref_B) @ self.w, self.b)
        }"""


def add_graph_idx(input):  # segment ids
    n = input['num_nodes']  # 30
    N = tf.shape(n)[0]  # 30 = len(n)
    graph_idx = tf.repeat(tf.range(N), n)
    return {
        **input,
        "graph_idx": graph_idx,
        "graph_count": N}


class Pooling(keras.layers.Layer, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def call(self, input):
        input = add_graph_idx(input)
        X = input['X']
        graph_idx = input["graph_idx"]
        graph_count = input["graph_count"]
        return tf.math.unsorted_segment_mean(X, graph_idx, graph_count)


"""4.1 Data Exchange Ops (API Level 2)
TF-GNN sends data across the graph as follows. Broadcasting from a node set to an edge set returns
for each edge the value from the specified endpoint (say, its source node). Pooling from an edge set
to a node set returns for each node the specified aggregation (sum, mean, max, etc.) of the value on
edges that have the node as the specified endpoint (say, their target node.) The tensors involved are
shaped like features of the respective node/edge set in the GraphTensor and can, but need not, be
stored in it. Similarly, graph context values can be broadcast to or pooled from the nodes or edges of
each graph component in a particular a node set or edge set.
Unlike multiplication with an adjacency matrix, this approach provides a natural place to insert peredge computations with one or more values, such as computing attention weights [33, 7], integrating
edge features into messages between nodes [12], or maintaining hidden states on edges [5]. array1 = [2, 3, 5]
tf.repeat(tf.range(3))
"""
