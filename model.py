from ogb.graphproppred import GraphPropPredDataset
import tensorflow as tf
import numpy as np
import keras
import convo
import convL
import networkx as nx

inputs = {'X': keras.Input(shape=(1,), dtype=tf.float32, name='X'),
          # Graphstruktur Platzhalter tensor
          'ref_A': keras.Input((), dtype=tf.int32, name='ref_A'),
          'ref_B': keras.Input((), dtype=tf.int32, name='ref_B')}
output = conv = convL.ConvolutionLayer(1)(inputs)['X']


model = (keras.Model(inputs=tf.nest.flatten(inputs), outputs=output))
d = {'X': tf.reshape(tf.constant([12, 3, 534, 46, 5]), (-1, 1)),
     'ref_A': tf.constant([]), 'ref_B': tf.constant([])}
model(d, )
model.compile(loss=tf.keras.losses.MSE, optimizer=tf.keras.optimizers.Adam(learning_rate=0.1
                                                                           ),  metrics=[tf.keras.metrics.RootMeanSquaredError()])  # loss function for model

model(d,)


data = tf.data.Dataset.from_tensors((d, tf.constant(
    np.concatenate([3*d['X']], axis=1), dtype=tf.float32)))

model.fit(data, epochs=100, callbacks=[keras.callbacks.TensorBoard(
    log_dir='Logs/Completed', write_graph=True)])


###################################################NEW TASK###################################################
# load dataset from OGB Library-Agnostic Loader


dataset = GraphPropPredDataset(name='ogbg-molhiv')

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

# set i as an arbitrary index
i = 0
graph, label = dataset[i]  # graph: library-agnostic graph object

edges = tf.constant(graph['edge_index'])
print(edges)
print(edges[-1])


"""testGraph = nx.Graph()
nodes = tf.constnt(graph['node_feat'])
edges = tf.constant(graph['edge_index'])
testGraph.add_nodes_from(tf.constant(graph['node_feat']))
testGraph.add_edges_from(tf.constant(graph['edge_index']))
"""
