import tensorflow as tf
from tensorflow import keras
import numpy as np
import convo as c
import batching
import poolingL
import modularized as m


class GConvoLayer(keras.layers.Layer):
    def __init__(self, units, activation='relu', drop_type=None, p=0.5, eps=0):
        super(GConvoLayer, self).__init__()
        self.units = units
        self.activation = keras.activations.get(activation)
        self.drop_type = drop_type
        self.p = p
        self.eps = eps

    def get_config(self):
        config = super().get_config()
        return {
            **config,
            "p": self.p,
            "eps": self.eps,
            "activation": keras.activations.serialize(self.activation),
            "units": self.units,
            "drop_type": self.drop_type
        }

    def build(self, input_shape):
        self.w1 = self.add_weight(shape=(
            input_shape['X'][-1], self.units), initializer='GlorotUniform', trainable=True, name="w1")
        self.b1 = self.add_weight(shape=(self.units, ),
                                  initializer='Zeros', trainable=True, name="b1")
        self.w2 = self.add_weight(shape=(
            self.units, self.units), initializer='GlorotUniform', trainable=True, name="w2")
        self.b2 = self.add_weight(shape=(self.units, ),
                                  initializer='Zeros', trainable=True, name="b2")

    def call(self, inputs):
        drop_type = self.drop_type
        activation = self.activation
        p = self.p
        X = inputs['X']
        ref_A = inputs['ref_A']
        ref_B = inputs['ref_B']

        # return simple neighbourhood aggregation sum without normalization
        mask = None
        # node_mask = False --> ??

        if drop_type == 'DropOut':
            X = c.dropout_mask(p, (c.get_shape(X, False))) * X
        elif drop_type == 'NodeSampling':
            mask = c.dropout_mask(p, (c.get_shape(X, True)))
        elif drop_type == 'DropEdge':
            mask = c.dropout_mask(p, (c.get_edgedropshape(X, ref_A, True)))
        elif drop_type == ' GDC':
            mask = c.dropout_mask(p, (c.get_edgedropshape(X, ref_A, False)))

        conv_X = (1 + self.eps) * X + c.convolution(X, ref_A, ref_B, mask)
        result = activation(tf.nn.bias_add(conv_X@self.w1, self.b1))
        result = activation(tf.nn.bias_add(result@self.w2, self.b2))
        return {
            **inputs,
            'X': result
        }

        # MLP
