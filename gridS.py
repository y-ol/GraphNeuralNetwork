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
import funcy as fy 


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

def getLabelMask(set):
    entry_list = list(set)
    labels = list()
    for entry in entry_list:
        labels.append(entry[0]["label_mask"].numpy())
    return np.concatenate(labels, axis=0)

param_grid = {'num_layers': [1, 2, 3, 4, 5, 6],
              'learning_rate': [0.0001, 0.001, 0.01],
              'optimizer': ['Adam'],
              'regularization': [None, 'DropOut', 'NodeSampling', 'DropEdge', 'GDC'],
              'probability': [0.3, 0.5, 0.7],
              'activation': ['relu', 'sigmoid', 'tanh'],
              'units': [32, 64],
              'convo_type': ['gcn', 'gin']}

param_combos = list(ParameterGrid(param_grid))


def create_model(activation, convo_type, learning_rate,
                 num_layers, probability, regularization,  units, node_features, include_mask, loss, metrics, num_tasks,
                 **kwargs) -> keras.Model:
    input = m.create_input(node_features, num_tasks if include_mask else 0)
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
        config['metrics'] = [tf.metrics.AUC(name = 'auc'), tf.metrics.BinaryAccuracy(name = 'bin_accuracy'),
                             tf.metrics.Precision(name = 'precision'), tf.metrics.Recall(name = 'recall')]
        config['evaluator'] = g.Evaluator(name='ogbg-molhiv')
    elif dataset_name == 'ogbg-molpcba':
        config['num_tasks'] = 128
        config['loss'] = tf.keras.losses.BinaryCrossentropy(
            from_logits=True)
        config['metrics'] = [tf.metrics.BinaryAccuracy(name = 'bin_accuracy'),
                             tf.metrics.Precision(name ='precision'), tf.metrics.Recall(name = 'recall')]
        config['evaluator'] = g.Evaluator(name='ogbg-molpcba')
        config['include_mask'] = True
    elif dataset_name == 'ogbn-products':  # multiclass classification
        config['num_tasks'] = 1
        config['loss'] = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True)
        config['metrics'] = [tf.keras.metrics.CategoricalAccuracy(name ='categorical_accuracy'),
                             tf.keras.metrics.TopKCategoricalAccuracy(name = 'top5categorical_accuracy',k=5)]
        config['evaluator'] = n.Evaluator(name='ogbn-products')
        # no missing label values
    elif dataset_name == 'ogbn-proteins':  # binary classification
        config['num_tasks'] = 112
        config['loss'] = tf.keras.losses.BinaryCrossentropy(
            from_logits=True)
        config['metrics'] = [tf.metrics.AUC(name = 'auc'), tf.metrics.BinaryAccuracy(name = 'bin_accuracy'),
                             tf.metrics.Precision(name = 'precision'), tf.metrics.Recall(name = 'recall')]
        config['evaluator'] = n.Evaluator(name='ogbn-proteins')
        # no missing label values
    elif dataset_name == 'ogbn-arxiv':  # multiclass classification
        config['num_tasks'] = 1
        config['loss'] = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True)
        config['metrics'] = [tf.metrics.CategoricalAccuracy(name='categorical_accuracy'),
                             tf.metrics.TopKCategoricalAccuracy(name = 'top20categorical_accuracy', k=20)]
        config['evaluator'] = n.Evaluator(name='ogbn-arxiv')
        # no missing label values
    elif dataset_name == 'ogbg-molesol' or dataset_name == 'ogbg-molfreesolv' or dataset_name == 'ogbg-mollipo': 
        config['num_tasks'] = 1 
        config['loss'] = tf.keras.losses.MSE
        config['metrics'] = [tf.keras.metrics.MeanAbsoluteError(name = 'MAE')]
        config['evaluator'] = None 
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    return config

def check_file_existence(folder_path, subfolder_name, filename):
    subfolder_path = os.path.join(folder_path, subfolder_name)
    file_path = os.path.join(subfolder_path, filename)
    return os.path.exists(file_path)

def train_and_evaluate(hyperparams, dataset_name, epochs = 200, experiment_results_dir='.', num_repeats=3, filter = None):
    # Load dataset
    if dataset_name == 'ogbg-molhiv' or dataset_name == 'ogbg-molpcba' or dataset_name == 'ogbg-molesol' or dataset_name == 'ogbg-molfreesolv' or dataset_name == 'ogbg-mollipo':
        dataset = GraphPropPredDataset(name=dataset_name)
    else:
        dataset = NodePropPredDataset(name=dataset_name)

    ds_config = config(dataset_name=dataset_name)
    evaluator = ds_config['evaluator']
    tfds = b.make_tf_datasets(dataset, **ds_config)
    training_batch = tfds['train']
    validation_batch = tfds['valid']
    test_data = tfds['test']

    # Set up directory structure for experiment results
    dataset_dir = os.path.join(experiment_results_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    hyperparams_path = os.path.join(dataset_dir, 'hyperparams.json')
    if not os.path.exists(hyperparams_path):
        with open(hyperparams_path, 'w') as f:
            # Write hyperparameters list to JSON file
            json.dump(hyperparams, f)

    # Iterate over hyperparameter configurations
    for i, hyperparams_dict in enumerate(hyperparams):
        if filter is not None and not filter(hyperparams_dict):
            continue

        # Train and evaluate the model
        print(
            f"Evaluating hyperparameter configuration {i} for dataset {dataset_name}.")

        # Repeat the experiment for the specified number of times
        for repeat in range(num_repeats):

            # Check if the repeat_i.json file exists
            repeat_filename = f"repeat_{repeat}.json"
            repeat_filepath = os.path.join(
                dataset_dir, f"hpconfig_{i}", repeat_filename)
            if os.path.exists(repeat_filepath):
                print(
                    f"Repeat {repeat} for hyperparameter configuration {i} already evaluated for dataset {dataset_name}.")
                continue
            model = create_model(
                node_features=dataset.graphs[0]['node_feat'].shape[-1], **ds_config, **hyperparams_dict)

            # Define callback and eval metrics
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=50, restore_best_weights=True)
            callback = [early_stopping]
            history = model.fit(
                training_batch, validation_data=validation_batch, epochs=epochs, callbacks=[callback])
            test_metrics = model.evaluate(test_data, return_dict=True)
            test_metrics_keras = dict() 
            for key, value in test_metrics.items(): 
                test_metrics_keras['test_'+key] = value
            test_predictions = model.predict(test_data)  # for ogb evaluator
            tf.keras.backend.clear_session()
            y_true_test = getLabels(test_data)
            if ds_config["include_mask"]:
                label_mask_test = getLabelMask(test_data)
                label_idxs_test = np.where(label_mask_test)
                full_y_true_test = np.full(label_mask_test.shape, np.nan, dtype=np.float32)
                full_y_pred_test = np.full(label_mask_test.shape, np.nan, dtype=np.float32)
                full_y_true_test[label_idxs_test] = y_true_test.reshape(-1)
                full_y_pred_test[label_idxs_test] = test_predictions.reshape(-1)
                input_dict = {"y_true": full_y_true_test,
                              "y_pred": full_y_pred_test}
            else:
                test_predictions = test_predictions.reshape(y_true_test.shape)
                input_dict = {"y_true": y_true_test, "y_pred": test_predictions}
            if evaluator is not None: 
                ogb_eval_result_dict = evaluator.eval(input_dict) 
                metric = fy.first(ogb_eval_result_dict.keys())

            result_dict = {
                'hyperparams': hyperparams_dict,
                **test_metrics_keras,
                # 'test_rocauc': test_rocauc_value,
                'training_history': history.history,
                'train_loss': history.history['loss'][early_stopping.best_epoch],
                'val_loss': history.history['val_loss'][early_stopping.best_epoch]}
            if evaluator is not None: 
                result_dict['test_' + metric]=  ogb_eval_result_dict[metric]
                
            # Write results to JSON file
            hyperparams_dir = os.path.join(dataset_dir, f"hpconfig_{i}")
            os.makedirs(hyperparams_dir, exist_ok=True)
            repeat_filename = f"repeat_{repeat}.json"
            repeat_filepath = os.path.join(hyperparams_dir, repeat_filename)
            with open(repeat_filepath, 'w') as f:
                json.dump(result_dict, f)
