from aim import Run
import tensorflow as tf
import pandas as pd
import numpy as np
import modularized as m
import batching as b
from sklearn.model_selection import ParameterGrid
from ogb.graphproppred import Evaluator
from ogb.nodeproppred import Evaluator
from ogb.graphproppred import GraphPropPredDataset
from ogb.nodeproppred import NodePropPredDataset
import json
import os


callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=50)


# aim_test.py

run = Run()

# set training hyperparameters
run['hparams'] = {
    'learning_rate': 0.001,
    'batch_size': 32,
}

# log metric
for i in range(10):
    run.track(i, name='numbers')

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


def create_grid_models(sample, dataset_name, mask=False):
    if dataset_name == 'ogbg-molhiv' or dataset_name == 'ogbg-molpcba':
        dataset = GraphPropPredDataset(name=dataset_name)
    else:
        dataset = NodePropPredDataset(name=dataset_name)
    training_batch = b.make_tf_datasets(dataset,)
    losses = loss_metric(dataset_name=dataset_name)['losses']
    metrics = loss_metric(dataset_name=dataset_name)['metrics']
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
        if True:  # check if already in txt file

            model = create_model(activation, convo_type, learning_rate,
                                 num_layers, optimizer, probability, regularization, units, dataset, mask)
            model.fit(training_batch, epochs=100, callbacks=[callback])
            # model.evaluate(b.make_tf_datasets['valid'])

        else:
            continue

    return model
