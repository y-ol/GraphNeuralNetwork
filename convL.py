import tensorflow as tf
import keras
import numpy as np
import convo as c


class ConvolutionLayer(keras.layers.Layer):
    def __init__(self, units, activation='relu', drop_type=None, p=0.5):
        super(ConvolutionLayer, self).__init__()
        self.units = units
        self.activation = keras.activations.get(activation)
        self.drop_type = drop_type
        self.p = p

    def get_config(self):
        config = super().get_config()

    def build(self, input_shape):

        self.w = self.add_weight(shape=(input_shape['X'][-1], self.units),
                                 initializer='GlorotUniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='GlorotUniform',
                                 trainable=True)

    def call(self, inputs):
        drop_type = self.drop_type
        activation = self.activation
        p = self.p
        X = inputs['X']
        ref_A = inputs['ref_A']
        ref_B = inputs['ref_B']

        mask = 1
        conv_X = (c.normalization(
            X + c.convolution(c.normalization(X, ref_A, ref_B, mask), ref_A, ref_B, mask), ref_A, ref_B, mask))
        result = activation(tf.nn.bias_add(conv_X @ self.w, self.b))

        if drop_type is None:
            return {
                **inputs,
                'X': result
            }
        elif drop_type == 'DropOut':
            return {
                **inputs,
                'X': c.dropout_mask(p, (c.get_shape(result, False))) * result


            }
        elif drop_type == 'NodeSampling':
            return {
                **inputs,
                'X': c.dropout_mask(p, (c.get_shape(result, True))) * result
            }
        elif drop_type == 'DropEdge':
            mask = c.dropout_mask(p, (c.get_edgedropshape(X, ref_A, True)))
            return {
                **inputs,
                'X': result
            }
        elif drop_type == 'GDC':
            mask = c.dropout_mask(p, (c.get_edgedropshape(X, ref_A, False)))
            return {
                **inputs,
                'X': result
            }
