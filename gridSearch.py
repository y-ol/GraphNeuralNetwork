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

# def build_run_index():
#     repo = Repo(".")
#     repo.list_all_runs


def create_grid_models_OLD(sample, dataset_name):
    if dataset_name == 'ogbg-molhiv' or dataset_name == 'ogbg-molpcba':
        dataset = GraphPropPredDataset(name=dataset_name)
    else:
        dataset = NodePropPredDataset(name=dataset_name)

    tfds = b.make_tf_datasets(dataset)
    training_batch = tfds['train']
    validation_batch = tfds['valid']
    test_data = tfds['test']
    # params for callback ?
    ds_config = config(dataset_name=dataset_name)
    evaluator = ds_config['evaluator']
    repo = Repo('/home/olga/GraphNeuralNetwork')

   
    for entry in sample:
        # TODO: unique ID for each run based on hparams
        experiment_id = dataset_name + "-" + getId(entry)
        # TODO: check if already database
        run = Run(repo = '/home/olga/GraphNeuralNetwork',
                  experiment=str(experiment_id))
        if True:
            # experiment not in the database, proceed with creating the run 
            print(f"[{experiment_id}] Evaluating hparams: {entry}")
            model = create_model(node_features=dataset.graphs[0]['node_feat'].shape[-1],
                                 **ds_config, **entry)
            run = Run(repo='/home/olga/GraphNeuralNetwork',
                      experiment=str(experiment_id))
            for key, value in entry.items():
                run[key] = value
            # moved callback bc of reference before assignment 
            callback = [tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=50), aim.tensorflow.AimCallback(
                repo='/home/olga/GraphNeuralNetwork', experiment=str(experiment_id))]

            model.fit(training_batch, validation_data=validation_batch, epochs=1,
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

            run.close()
        else:
            continue

    return 1



def create_grid_models_first(sample, dataset_name):
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

    results = {}

    for entry in sample:
        # TODO: unique ID for each run based on hparams
        experiment_id = dataset_name + "-" + getId(entry)
        if experiment_id in results:
            continue 

        # experiment not in the database, proceed with creating the run
        print(f"[{experiment_id}] Evaluating hparams: {entry}")
        model = create_model(node_features=dataset.graphs[0]['node_feat'].shape[-1],
                                 **ds_config, **entry)
       
        # moved callback bc of reference before assignment
        callback = [tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=50)]

        model.fit(training_batch, validation_data=validation_batch, epochs=1,
                      callbacks=[callback])
        test_predictions = model.predict(test_data)  # für ogb evaluator
        # evaluate using OGBEvaluator
        input_dict = {
            "y_true": getLabels(test_data), "y_pred": test_predictions}
        result_dict = evaluator.eval(input_dict)
        test_acc_value = result_dict["rocauc"]

        test_acc_value = model.evaluate(test_data)
        results[experiment_id] = {
            'hyperparams': entry, 'test_rocauc': test_acc_value}

    with open('hyperparams.json', 'w') as f:
        json.dump(results, f)

    return 1


# Overrides JSON file every time it is executed

def create_grid_models(sample, dataset_name):
    if dataset_name == 'ogbg-molhiv' or dataset_name == 'ogbg-molpcba':
        dataset = GraphPropPredDataset(name=dataset_name)
    else:
        dataset = NodePropPredDataset(name=dataset_name)

    tfds = b.make_tf_datasets(dataset)
    training_batch = tfds['train']
    validation_batch = tfds['valid']
    test_data = tfds['test']

    # load or create hyperparameters file
    hyperparams_file = f"experiment_results/{dataset_name}/hyperparams.json"
    if os.path.exists(hyperparams_file):
        with open(hyperparams_file, "r") as f:
            hyperparams_dict = json.load(f)
    else:
        hyperparams_dict = {}

    # loop through sample and create model runs
    for entry in sample:
        hp_config_id = str(getId(entry))
        hp_config_dir = f"experiment_results/{dataset_name}/hpconfig_{hp_config_id}"

        # check if hyperparameter config has already been evaluated
        if hp_config_id in hyperparams_dict:
            print(f"Skipping hpconfig {hp_config_id}, already evaluated.")
            continue

        # create new hyperparameter config and model runs
        print(f"Evaluating hpconfig {hp_config_id}")
        os.makedirs(hp_config_dir, exist_ok=True)

        # create a JSON file to hold the hyperparameters
        with open(f"{hp_config_dir}/hyperparams.json", "w") as f:
            json.dump(entry, f)

        # TODO: create and fit the model, evaluate, and write results to JSON

        # add the evaluated hyperparameter config to the hyperparameters file
        hyperparams_dict[hp_config_id] = entry
        with open(hyperparams_file, "w") as f:
            json.dump(hyperparams_dict, f)

    return 1
