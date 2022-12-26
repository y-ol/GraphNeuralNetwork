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
        return {
          **config,
          "p": self.p,
          "activation": keras.activations.serialize(self.activation),
          "units": self.units,
          "drop_type": self.drop_type
        }

    def build(self, input_shape):

        self.w = self.add_weight(shape=(input_shape['X'][-1], self.units),
                                 initializer='GlorotUniform',
                                 trainable=True, name="w")
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='Zeros',
                                 trainable=True, name="b")

    def call(self, inputs):
        drop_type = self.drop_type
        activation = self.activation
        p = self.p
        X = inputs['X']
        ref_A = inputs['ref_A']
        ref_B = inputs['ref_B']

        mask = None
        node_mask = False
        if drop_type == 'DropOut':
            X = c.dropout_mask(p, (c.get_shape(X, False))) * X
        elif drop_type == 'NodeSampling':
            mask = c.dropout_mask(p, (c.get_shape(X, True)))
            node_mask = True
        elif drop_type == 'DropEdge':
            mask = c.dropout_mask(p, (c.get_edgedropshape(X, ref_A, True)))
        elif drop_type == 'GDC':
            mask = c.dropout_mask(p, (c.get_edgedropshape(X, ref_A, False)))

        conv_X = c.normalize_convo(X, ref_A, ref_B, mask, node_mask)

        return {
            **inputs,
            'X': activation(tf.nn.bias_add(conv_X@self.w, self.b))
        }
