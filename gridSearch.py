import aim
from aim import Repo
from aim.tensorflow import AimCallback
from aim import Run
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import numpy as np
import modularized as m
import batching as b
from sklearn.model_selection import ParameterGrid
import ogb.graphproppred as g
import ogb.nodeproppred as n
from ogb.graphproppred import GraphPropPredDataset
from ogb.nodeproppred import NodePropPredDataset
import json
import os
from keras import callbacks


def getLabels(set):
    entry_list = list(set)
    labels = list()
    for entry in entry_list:
        labels.append(entry[1].numpy())
    return np.concatenate(labels, axis=0)

# metrics to string ??


param_grid = {'num_layers': [1, 2, 3, 4, 5, 6],
              'learning_rate': [0.0001, 0.001, 0.01],
              'optimizer': ['Adam'],
              'regularization': ['DropOut', 'NodeSampling', 'DropEdge', 'GDC'],
              'probability': [0.3, 0.5, 0.7],
              'activation': ['relu', 'sigmoid', 'tanh'],
              'units': [32, 64],
              'convo_type': ['gcn', 'gin']}

param_combos = list(ParameterGrid(param_grid))


def create_model(activation, convo_type, learning_rate, num_layers, optimizer, probability, regularization,  units, dataset, mask, **kwargs) -> keras.Model:
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


def config(dataset_name):
    config = dict()
    if dataset_name == 'ogbg-molhiv':  # binary classification
        config['num_tasks'] = 1
        config['losses'] = [tf.keras.losses.BinaryCrossentropy(
            from_logits=True), tf.keras.losses.Hinge()]
        config['metrics'] = [tf.metrics.AUC, tf.metrics.BinaryAccuracy,
                             tf.metrics.Precision, tf.metrics.Recall]
        config['evaluator'] = g.Evaluator(name='ogbg-molhiv')
    elif dataset_name == 'ogbg-molpcba':
        config['num_tasks'] = 128
        config['losses'] = [tf.keras.losses.BinaryCrossentropy(
            from_logits=True), tf.keras.losses.Hinge()]
        config['metrics'] = [tf.metrics.BinaryAccuracy,
                             tf.metrics.Precision, tf.metrics.Recall]
        config['evaluator'] = g.Evaluator(name='ogbg-molpcba')
    elif dataset_name == 'ogbn-products':  # multiclass classification
        config['num_tasks'] = 1
        config['losses'] = [tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                            tf.keras.losses.CategoricalHinge()]
        config['metrics'] = [tf.keras.metrics.CategoricalAccuracy(),
                             tf.keras.metrics.TopKCategoricalAccuracy(k=5)]
        config['evaluator'] = n.Evaluator(name='ogbn-products')
    elif dataset_name == 'ogbn-proteins':  # binary classification
        config['num_tasks'] = 112
        config['losses'] = [tf.keras.losses.BinaryCrossentropy(
            from_logits=True), tf.keras.losses.Hinge()]
        config['metrics'] = [tf.metrics.AUC, tf.metrics.BinaryAccuracy,
                             tf.metrics.Precision, tf.metrics.Recall]
        config['evaluator'] = n.Evaluator(name='ogbn-proteins')
    elif dataset_name == 'ogbn-arxiv':  # multiclass classification
        config['num_tasks'] = 1
        config['losses'] = [tf.keras.losses.CategoricalCrossentropy(
            from_logits=True), tf.keras.losses.CategoricalHinge()]
        config['metrics'] = [tf.metrics.CategoricalAccuracy,
                             tf.metrics.TopKCategoricalAccuracy(k=20)]
        config['evaluator'] = n.Evaluator(name='ogbn-arxiv')
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    return config


def create_grid_models(sample, dataset_name, mask=False):
    if dataset_name == 'ogbg-molhiv' or dataset_name == 'ogbg-molpcba':
        dataset = GraphPropPredDataset(name=dataset_name)

    else:
        dataset = NodePropPredDataset(name=dataset_name)
    experiment_number = 0
    training_batch = b.make_tf_datasets(dataset)['train']
    validation_batch = b.make_tf_datasets(dataset)['valid']
    test_data = b.make_tf_datasets(dataset)['test']
    # params for callback ?
    losses = config(dataset_name=dataset_name)['losses']
    metrics = config(dataset_name=dataset_name)['metrics']
    evaluator = config(dataset_name=dataset_name)['evaluator']

    callback = [tf.keras.callbacks.EarlyStopping(
        monitor='accuracy,loss,rocauc', patience=50), aim.tensorflow.AimCallback(
        repo='/home/olga/GraphNeuralNetwork', experiment=str(experiment_number))]
    for entry in sample:
        if True:  # check if already database

            model = create_model(dataset=dataset, mask=mask, **entry)
            run = Run(repo='/home/olga/GraphNeuralNetwork',
                      experiment=str(experiment_number))
            run['hparams'] = {'num_layers': entry['num_layers'],
                              'learning_rate': entry['learning_rate'],
                              'optimizer': ['Adam'],
                              'regularization': entry['regularization'],
                              'probability': entry['probability'],
                              'activation': entry['activation'],
                              'units': entry['units'],
                              'convo_type': entry['convo_type']}

            model.fit(training_batch, validation_data=validation_batch, epochs=3,
                      callbacks=[callback])
            test_predictions = model.predict(test_data)  # für ogb evaluator
            # evaluate using OGBEvaluator
            input_dict = {
                "y_true": getLabels(test_data), "y_pred": test_predictions}
            result_dict = evaluator.eval(input_dict)
            test_acc_value = result_dict["rocauc"]
            run.track({'test_rocauc': test_acc_value})
            test_acc_value = model.evaluate(test_data)
            run.track({'accuracy': test_acc_value})

            experiment_number += 1
            run.close()
        else:
            continue

    return 1


# test_predictions = model.predict(test_data)
# test_labels = getLabels(test_data)
# test_results = model.evaluate(test_data, metrics=metrics)
# test_acc_value = test_results[0]  # binary accuracy
# test_precision_value = test_results[1]  # precision
# test_recall_value = test_results[2]  # recall
# test_rocauc_value = test_results[3]  # ROC-AUC

# run.track({'test_accuracy': test_acc_value, 'test_precision': test_precision_value,
#            'test_recall': test_recall_value, 'test_rocauc': test_rocauc_value})
