import tensorflow as tf
import pandas as pd
import numpy as np
import modularized as m
import batching as b
from sklearn.model_selection import ParameterGrid

param_grid = {'num_layers': [1, 2, 3, 4, 5, 6],
              'learning_rate': [0.0001, 0.001, 0.01],
              'optimizer': ['Adam'],
              'regularization': ['DropOut', 'NodeSampling', 'DropEdge', 'GDC'],
              'probability': [0.3, 0.5, 0.7],
              'activation': ['relu', 'sigmoid', 'tanh'],
              'units': [32, 64],
              'convo_type': ['gcn', 'gin']}

param_combos = list(ParameterGrid(param_grid))


def create_model(activation, convo_type, learning_rate, num_layers, optimizer, probability, regularization,  units, dataset, mask):
    input = m.create_input(dataset, mask)
    layer_list = list()
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    for i in range(num_layers):
        if convo_type == 'gcn':
            layer_list.append('Convolution')
        else:
            layer_list.append('ginConvo')
    layer_list += ['Pooling', 'Dense']

    model = m.create_model(layer_list, input, regularization,
                           probability, units, activation)

    # if/else classification/regression for loss&metrics
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

    return model


def create_list_of_models(sample):  # sample = param_combos)
    list_models = list()
    for entry in sample:
        activation = entry['activation']
        convo_type = entry['convo_type']
        learning_rate = entry['learning_rate']
        num_layers = entry['num_layers']
        optimizer = entry['optimizer']
        probability = entry['probability']
        regularization = entry['regularization']
        units = entry['units']
        list_models.append(g.create_model(activation, convo_type, learning_rate,
                           num_layers, optimizer, probability, regularization, units))

    return list_models

# main


callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=50)


def models_fit(list_models, inputs):
    for i in list_models:
        list_models[i].fit(inputs['train'].cache(),
                           epochs=100, callbacks=callback)
