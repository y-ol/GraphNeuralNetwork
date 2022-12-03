from ogb.graphproppred import GraphPropPredDataset
import tensorflow as tf
import numpy as np
from tensorflow import keras
import convo
import convL
import networkx as nx
import batching
import poolingL


inputs = {'X': keras.Input(shape=(9,), dtype=tf.float32, name='X'),
          'ref_A': keras.Input((), dtype=tf.int32, name='ref_A'),
          'ref_B': keras.Input((), dtype=tf.int32, name='ref_B'),
          'num_nodes': keras.Input((), dtype=tf.int32, name='num_nodes')}


# Mehrere ConvL 3Layer, units = 32 ,

# Dense Layer
conv1 = convL.ConvolutionLayer(32)(inputs)
conv2 = convL.ConvolutionLayer(32)(conv1)
conv3 = convL.ConvolutionLayer(32)(conv2)
pooling1 = poolingL.Pooling()(conv3)  # 32 dim Feature -Vektor

dense1 = keras.layers.Dense(units=32, activation='relu')(pooling1)
outputs = keras.layers.Dense(units=1)(dense1)

model = (keras.Model(inputs=tf.nest.flatten(inputs), outputs=outputs))
