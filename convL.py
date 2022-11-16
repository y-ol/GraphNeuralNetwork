import tensorflow as tf
import keras
import numpy as np
import convo as c


class ConvolutionLayer(keras.layers.Layer):
    def __init__(self, units):
        super(ConvolutionLayer, self).__init__()
        self.units = units

    def get_config(self):
        config = super().get_config()

    # Input_shape = shape of the nodes of the input Graph input: 4D -> output: 4D
    def build(self, input_shape):

        self.w = self.add_weight(shape=(input_shape['X'][-1], self.units),
                                 initializer='GlorotUniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='GlorotUniform',
                                 trainable=True)

    def call(self, inputs):
        X = inputs['X']
        ref_A = inputs['ref_A']
        ref_B = inputs['ref_B']

        return {
            'X': tf.nn.bias_add(c.convolution(X, ref_A, ref_B) @ self.w, self.b),
            'ref_A': inputs['ref_A'],
            'ref_B': inputs['ref_B']
        }
