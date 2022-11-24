from ogb.graphproppred import GraphPropPredDataset
import tensorflow as tf
import numpy as np
from tensorflow import keras
import convo
import convL
import networkx as nx
import batching
import poolingL

inputs = {'X': keras.Input(shape=(9,), dtype=tf.float32, name='X'),  # batch implicit dynamic
          # Graphstruktur Platzhalter tensor
          'ref_A': keras.Input((), dtype=tf.int32, name='ref_A'),
          'ref_B': keras.Input((), dtype=tf.int32, name='ref_B'),
          'num_nodes': keras.Input((), dtype=tf.int32, name='num_nodes')}
# 9 is input - (inputs)['X'] - def instance of convoLution class
conv = convL.ConvolutionLayer(1)(
    inputs)  # (1) dimension pro Node
output = poolingL.Pooling()(conv)

model = (keras.Model(inputs=tf.nest.flatten(inputs), outputs=output))

model.compile(loss=tf.keras.losses.MSE, optimizer=tf.keras.optimizers.Adam(learning_rate=0.1
                                                                           ),  metrics=[tf.keras.metrics.RootMeanSquaredError()])  # loss function for model


# data = tf.data.Dataset.from_tensors((d, tf.constant(
# np.concatenate([3*d['X']], axis=1), dtype = tf.float32)))
"""data = tf.data.Dataset.from_tensors((
    batching.first_batch['X'], batching.first_batch['ref_A'], batching.first_batch['ref_B']), axis=1, dtype=tf.float32)"""
data = tf.data.Dataset.from_tensors(
    (batching.first_batch, batching.labels[:30]))
# Hier zu dem Input kompatible daten
model.fit(data, epochs=100, callbacks=[keras.callbacks.TensorBoard(
    log_dir='Logs/Completed', write_graph=True)])
model.predict(data)
