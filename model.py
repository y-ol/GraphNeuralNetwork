from ogb.graphproppred import GraphPropPredDataset
import tensorflow as tf
import numpy as np
from tensorflow import keras
import convo
import convL
import networkx as nx
import batching
import poolingL
import gin

inputs = {'X': keras.Input(shape=(9,), dtype=tf.float32, name='X'),
          'ref_A': keras.Input((), dtype=tf.int32, name='ref_A'),
          'ref_B': keras.Input((), dtype=tf.int32, name='ref_B'),
          'num_nodes': keras.Input((), dtype=tf.int32, name='num_nodes')}


conv1 = convL.ConvolutionLayer(32)(inputs)
conv2 = convL.ConvolutionLayer(32)(conv1)
conv3 = convL.ConvolutionLayer(32)(conv2)
pooling1 = poolingL.Pooling()(conv3)  # 32 dim Feature -Vektor

dense1 = keras.layers.Dense(units=32, activation='relu')(pooling1)
outputs = keras.layers.Dense(units=1)(dense1)

model = (keras.Model(inputs=tf.nest.flatten(inputs), outputs=outputs))


# new models, since colab bug
"""
c1 = convL.ConvolutionLayer(32)(inputs)
p1 = poolingL.Pooling()(c1)
d1= keras.layers.Dense(units = 32, activation = 'relu')(p1)
out = keras.layers.Dense(units =1)(p1)
model2 = (keras.Model(inputs= tf.nest.flatten(inputs), outputs= out))
"""
