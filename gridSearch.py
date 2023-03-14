import tensorflow as tf
import pandas as pd
import numpy as np
import modularized as m
import batching as b
from sklearn.model_selection import ParameterGrid
from ogb.graphproppred import Evaluator
from ogb.nodeproppred import Evaluator
import json
import os


callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=50)

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

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

    return model


def loss_metric(dataset_name):
    config = dict()
    if dataset_name == 'ogbg-molhiv':  # binary classification
        config['num_tasks'] = 1
        config['losses'] = [tf.keras.losses.BinaryCrossentropy(
            from_logits=True), tf.keras.losses.Hinge()]
        config['metrics'] = [tf.metrics.AUC, tf.metrics.BinaryAccuracy,
                             tf.metrics.Precision, tf.metrics.Recall]
    elif dataset_name == 'ogbg-molpcba':
        config['num_tasks'] = 128
        config['losses'] = [tf.keras.losses.BinaryCrossentropy(
            from_logits=True), tf.keras.losses.Hinge()]
        config['metrics'] = [tf.metrics.BinaryAccuracy,
                             tf.metrics.Precision, tf.metrics.Recall]
    elif dataset_name == 'ogbn-products':  # multiclass classification
        config['num_tasks'] = 1
        config['losses'] = [tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                            tf.keras.losses.CategoricalHinge()]
        config['metrics'] = [tf.keras.metrics.CategoricalAccuracy(),
                             tf.keras.metrics.TopKCategoricalAccuracy(k=5)]
    elif dataset_name == 'ogbn-proteins':  # binary classification
        config['num_tasks'] = 112
        config['losses'] = [tf.keras.losses.BinaryCrossentropy(
            from_logits=True), tf.keras.losses.Hinge()]
        config['metrics'] = [tf.metrics.AUC, tf.metrics.BinaryAccuracy,
                             tf.metrics.Precision, tf.metrics.Recall]
    elif dataset_name == 'ogbn-arxiv':  # multiclass classification
        config['num_tasks'] = 1
        config['losses'] = [tf.keras.losses.CategoricalCrossentropy(
            from_logits=True), tf.keras.losses.CategoricalHinge()]
        config['metrics'] = [tf.metrics.CategoricalAccuracy,
                             tf.metrics.TopKCategoricalAccuracy(k=20)]
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    return config


# JSON reader

def read_contents(json_filename):


    # Read contents of JSON file into a dictionary
with open("my_file.json", "r") as f:
    data = json.load(f)

# Check if a specific entry exists in the dictionary
if "my_key" in data:
    print("Found entry:", data["my_key"])
else:
    print("Entry not found.")


def create_grid_models(sample, dataset, name_dataset, mask=False):
    #model_idx = list()
    tracking = dict()
    losses = loss_metric(dataset_name=name_dataset)['losses']
    metrics = loss_metric(dataset_name=name_dataset)['metrics']
    for entry in sample:
        activation = entry['activation']
        convo_type = entry['convo_type']
        learning_rate = entry['learning_rate']
        num_layers = entry['num_layers']
        optimizer = entry['optimizer']
        probability = entry['probability']
        regularization = entry['regularization']
        units = entry['units']
        param_set = [activation, convo_type, learning_rate,
                     num_layers, optimizer, probability, regularization, units]
        if param_set not in 'params.json':
            model = create_model(activation, convo_type, learning_rate,
                                 num_layers, optimizer, probability, regularization, units, dataset, mask)
            model.fit(dataset['train'].cashe(),
                      epochs=100, callbacks=[callback])
            model.evaluate(b.make_tf_datasets['valid'])
            tracking[entry] = entry
            # Write parameter combinations to JSON file
            with open("params.json", "w") as f:
                json.dump(tracking, f)
            # model_idx.append([activation, convo_type, learning_rate,
            # num_layers, optimizer, probability, regularization, units])

        else:
            continue

    return model
