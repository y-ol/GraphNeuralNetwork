import tensorflow as tf
import numpy as np
from tensorflow import keras
import convL
import poolingL
import gin

# First specify input --> no data


def create_input(dataset, include_mask=False):
    node_features = dataset.graphs[0]['node_feat'].shape[-1]
    inputs = {'X': keras.Input(shape=(node_features,), dtype=tf.float32, name='X'),
              'ref_A': keras.Input((), dtype=tf.int32, name='ref_A'),
              'ref_B': keras.Input((), dtype=tf.int32, name='ref_B'),
              'num_nodes': keras.Input((), dtype=tf.int32, name='num_nodes')}
    if include_mask:
        inputs['label_mask'] = keras.Input(
            node_features, dtype=tf.bool, name='label_mask')
    return inputs

# Inputs with values in batching


def create_model(list_layers, initial_input, units=32, activation='relu', output_units=1):
    output = initial_input
    for element in list_layers:
        if element == 'Convolution':
            output = convL.ConvolutionLayer(32, activation)(output)
        elif element == 'Pooling':
            output = poolingL.Pooling(activation)(output)
        elif element == 'Dense':
            output = keras.layers.Dense(units, activation)(output)
        elif element == 'ginConvo':
            output = gin.GConvoLayer(32, activation)(output)
    output = keras.layers.Dense(units=output_units)(output)
    if 'label_mask' in initial_input:
        output = tf.gather_nd(output, tf.where(
            initial_input['label_mask']))  # not real-world
    model = (keras.Model(inputs=tf.nest.flatten(initial_input),
             outputs=output))
    return model


# Node Prediction logic 2do
