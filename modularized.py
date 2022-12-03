import tensorflow as tf
import numpy as np
from tensorflow import keras
import convL
import poolingL


# First specify input --> no data
def specify_input(dictionary):
    input_spec = dict()
    for key, value in dictionary.items():
        input_spec[key] = keras.Input(shape=(tf.shape(tf.constant(value))), dtype=(
            tf.constant(value)).dtype, name=str(key))

    return input_spec

# Inputs with values in batching


def create_model(list_layers, initial_input, units=32, activation='relu'):
    output = initial_input
    for element in list_layers:
        if element == 'Convolution':
            output = convL.ConvolutionLayer(32, activation)(output)
        elif element == 'Pooling':
            output = poolingL.Pooling(activation)(output)
        elif element == 'Dense':
            output = keras.layers.Dense(units, activation)(output)

    model = (keras.Model(inputs=tf.nest.flatten(initial_input),
             outputs=keras.layers.Dense(units=1)(output)))
    return model
