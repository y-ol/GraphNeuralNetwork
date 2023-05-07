import aim
from aim import Repo
from aim.tensorflow import AimCallback
from aim import Run
from aim.sdk.run import Run
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


def getId(entry):
    id = ''
    for key, value in entry.items():
        id += key + ':' + str(value) + " "
    return id


def getLabels(set):
    entry_list = list(set)
    labels = list()
    for entry in entry_list:
        labels.append(entry[1].numpy())
    return np.concatenate(labels, axis=0)


param_grid = {'num_layers': [1, 2, 3, 4, 5, 6],
              'learning_rate': [0.0001, 0.001, 0.01],
              'optimizer': ['Adam'],
              'regularization': ['DropOut', 'NodeSampling', 'DropEdge', 'GDC'],
              'probability': [0.3, 0.5, 0.7],
              'activation': ['relu', 'sigmoid', 'tanh'],
              'units': [32, 64],
              'convo_type': ['gcn', 'gin']}

param_combos = list(ParameterGrid(param_grid))


def create_model(activation, convo_type, learning_rate,
                 num_layers, probability, regularization,  units, node_features, include_mask, loss, metrics, num_tasks,
                 **kwargs) -> keras.Model:
    input = m.create_input(node_features, include_mask)
    layer_list = list()
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    for i in range(num_layers):
        if convo_type == 'gcn':
            layer_list.append('Convolution')
        else:
            layer_list.append('ginConvo')
    layer_list += ['Pooling', 'Dense']

    model: keras.Model = m.create_model(layer_list, input, regularization,
                                        probability, units, activation, output_units=num_tasks)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

# TODO: click

def config(dataset_name):
    config = dict(include_mask=False)
    if dataset_name == 'ogbg-molhiv':  # binary classification
        config['num_tasks'] = 1
        config['loss'] = tf.keras.losses.BinaryCrossentropy(
            from_logits=True)
        config['metrics'] = [tf.metrics.AUC(), tf.metrics.BinaryAccuracy(),
                             tf.metrics.Precision(), tf.metrics.Recall()]
        config['evaluator'] = g.Evaluator(name='ogbg-molhiv')
    elif dataset_name == 'ogbg-molpcba':
        config['num_tasks'] = 128
        config['loss'] = tf.keras.losses.BinaryCrossentropy(
            from_logits=True)
        config['metrics'] = [tf.metrics.BinaryAccuracy(),
                             tf.metrics.Precision(), tf.metrics.Recall()]
        config['evaluator'] = g.Evaluator(name='ogbg-molpcba')
        config['include_mask'] = True
    elif dataset_name == 'ogbn-products':  # multiclass classification
        config['num_tasks'] = 1
        config['loss'] = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True)
        config['metrics'] = [tf.keras.metrics.CategoricalAccuracy(),
                             tf.keras.metrics.TopKCategoricalAccuracy(k=5)]
        config['evaluator'] = n.Evaluator(name='ogbn-products')
        # no missing label values
    elif dataset_name == 'ogbn-proteins':  # binary classification
        config['num_tasks'] = 112
        config['loss'] = tf.keras.losses.BinaryCrossentropy(
            from_logits=True)
        config['metrics'] = [tf.metrics.AUC(), tf.metrics.BinaryAccuracy(),
                             tf.metrics.Precision(), tf.metrics.Recall()]
        config['evaluator'] = n.Evaluator(name='ogbn-proteins')
        # no missing label values
    elif dataset_name == 'ogbn-arxiv':  # multiclass classification
        config['num_tasks'] = 1
        config['loss'] = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True)
        config['metrics'] = [tf.metrics.CategoricalAccuracy(),
                             tf.metrics.TopKCategoricalAccuracy(k=20)]
        config['evaluator'] = n.Evaluator(name='ogbn-arxiv')
        # no missing label values
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    return config


def train_and_evaluate(hyperparams, dataset_name, experiment_results_dir='/home/olga/GraphNeuralNetwork'):
    # Load dataset
    if dataset_name == 'ogbg-molhiv' or dataset_name == 'ogbg-molpcba':
        dataset = GraphPropPredDataset(name=dataset_name)
    else:
        dataset = NodePropPredDataset(name=dataset_name)

    tfds = b.make_tf_datasets(dataset)
    training_batch = tfds['train']
    validation_batch = tfds['valid']
    test_data = tfds['test']
    ds_config = config(dataset_name=dataset_name)
    evaluator = ds_config['evaluator']
    
    # Set up directory structure for experiment results
    dataset_dir = os.path.join(experiment_results_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    hyperparams_path = os.path.join(dataset_dir, 'hyperparams.json')
    if os.path.exists(hyperparams_path):
        with open(hyperparams_path, 'r') as f:
            hyperparams_list = json.load(f)
    else:
        hyperparams_list = []

    # Iterate over hyperparameter configurations
    for i, hyperparams_dict in enumerate(hyperparams):
        # Check if this configuration has already been evaluated
        if hyperparams_dict in hyperparams_list:
            print(
                f"Hyperparameter configuration {i} already evaluated for dataset {dataset_name}.")
            continue

        # Train and evaluate the model
        print(
            f"Evaluating hyperparameter configuration {i} for dataset {dataset_name}.")
        model = create_model(
            node_features=dataset.graphs[0]['node_feat'].shape[-1], **ds_config, **hyperparams_dict)
        
        # Define callback  and eval metrics 
        callback = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=50)]
        history = model.fit(
            training_batch, validation_data=validation_batch, epochs=1, callbacks=[callback])
        test_loss, *metrics = model.evaluate(test_data)
        test_predictions = model.predict(test_data)  # for ogb evaluator
        input_dict = {"y_true": getLabels(
            test_data), "y_pred": test_predictions}
        result_dict = evaluator.eval(input_dict)
        test_rocauc_value = result_dict["rocauc"]

        # Write results to JSON file
        hyperparams_list.append(hyperparams_dict)
        hyperparams_dir = os.path.join(dataset_dir, f"hpconfig_{i}")
        os.makedirs(hyperparams_dir, exist_ok=True)
        repeat_filename = f"repeat_{len(os.listdir(hyperparams_dir))}.json"
        repeat_filepath = os.path.join(hyperparams_dir, repeat_filename)
        with open(repeat_filepath, 'w') as f:
            json.dump({
                'hyperparams': hyperparams_dict,
                'test_loss': test_loss,
                'test_acc': metrics,
                'test_rocauc': test_rocauc_value,
                'training_history': history.history
            }, f)

        # Write hyperparameters list to JSON file
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams_list, f)
