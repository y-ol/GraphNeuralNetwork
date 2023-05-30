import tensorflow as tf
import numpy as np
from tensorflow import keras
import convL
import poolingL
import gin

# Method for Model input specification


def create_input(node_features, mask_dim=0):
    inputs = {'X': keras.Input(shape=(node_features,), dtype=tf.float32, name='X'),
              'ref_A': keras.Input((), dtype=tf.int32, name='ref_A'),
              'ref_B': keras.Input((), dtype=tf.int32, name='ref_B'),
              'num_nodes': keras.Input((), dtype=tf.int32, name='num_nodes')}
    if mask_dim > 0:
        inputs['label_mask'] = keras.Input(
            mask_dim, dtype=tf.bool, name='label_mask')
    return inputs

# Inputs with values in batching


def create_model(list_layers, initial_input, drop_type, p, units=32, activation='relu', output_units=1):
    output = initial_input
    for element in list_layers:
        if element == 'Convolution':
            output = convL.ConvolutionLayer(
                units, activation, drop_type, p)(output)
        elif element == 'Pooling':
            output = poolingL.Pooling(activation)(output)
        elif element == 'Dense':
            output = keras.layers.Dense(units, activation)(output)
        elif element == 'ginConvo':
            output = gin.GConvoLayer(units, activation, drop_type, p)(output)
    output = keras.layers.Dense(units=output_units)(output)
    if 'label_mask' in initial_input:
        output = tf.gather_nd(output, tf.where(
            initial_input['label_mask']))  # not real-world
    model = (keras.Model(inputs=tf.nest.flatten(initial_input),
             outputs=output))
    return model


# Node Prediction logic 2do
# Units = 1 -> Regression
# Units = num_classes
