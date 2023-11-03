# import tensorflow as tf
import os
import csv
import pandas as pd
import json
import numpy as np
import statistics
import click
import cache
from sklearn.model_selection import ParameterGrid

results_dir = os.environ.get("RESULTS_DIR", "results")
directory = "."
datasets = ['ogbg-molhiv', 'ogbg-molpcba',
            'ogbg-molesol', 'ogbg-molfreesolv', 'ogbg-mollipo']


def to_camel_case(snake_str):
    return "".join(x.capitalize() for x in snake_str.lower().split("_"))


def to_lower_camel_case(snake_str):
    snake_str = str(snake_str)
    # We capitalize the first letter of each component except the first one
    # with the 'capitalize' method and join them together.
    camel_string = to_camel_case(snake_str)
    return snake_str[0].lower() + camel_string[1:]

# Get the folder paths --> use this method


def get_folder_paths(directory, ds_list):
    list_pf_paths = list()
    for entry in ds_list:
        list_pf_paths.append(os.path.join(directory, entry))
    return list_pf_paths

# lookup relevant metric name for dataset


def get_metric(dataset_name, val=False):
    metric = "val_" if val else "test_"

    if val:
        if dataset_name == 'ogbg-molpcba':
            metric += "loss"
        else:
            metric += "loss"
    else:
        if dataset_name == 'ogbg-molhiv':
            metric += 'rocauc'
        elif dataset_name == 'ogbg-molpcba':
            metric += 'ap'
        elif (dataset_name == 'ogbg-molesol' or dataset_name == 'ogbg-molfreesolv' or dataset_name == 'ogbg-mollipo'):
            metric += 'loss'  # assuming it's called mse in the keras evaluator
        else:
            raise Exception("dataset unknown")
    return metric


def min_or_max(metric):
    if ('loss' in metric):
        return min, np.Inf
    else:
        return max, -np.Inf


def find_subdirs(dir):
    return [
        entry.name
        for entry in os.scandir(dir)
        if entry.is_dir()
    ]


def add_metric_from_history(ds_directory, metric, mode=None):
    hpconfig_folders = find_subdirs(ds_directory)
    for j, hpconfig_folder in enumerate(hpconfig_folders):
        print(f"Fixing {j+1}/{len(hpconfig_folders)} ({mode=})")
        repeats_folder = os.path.join(ds_directory, hpconfig_folder)
        repeat_files = [entry.name for entry in os.scandir(
            repeats_folder) if entry.is_file() and entry.name.startswith('repeat_')]
        for repeat_file in repeat_files:
            repeat_path = os.path.join(repeats_folder, repeat_file)
            repeat_data = cache.read_cached_json(repeat_path)
            final_val_loss = repeat_data['val_loss']
            training_history = repeat_data['training_history']
            if mode == "max":
                final_metric = np.max(training_history[metric])
            elif mode == "min":
                final_metric = np.min(training_history[metric])
            else:
                best_i = None
                best_val = np.Inf
                for i, vl in enumerate(training_history['val_loss']):
                    diff = np.abs(final_val_loss - vl)
                    if diff < best_val:
                        best_i = i
                final_metric = training_history[metric][best_i]
            repeat_data[metric] = final_metric
            with open(repeat_path, "w") as f:
                json.dump(repeat_data, f)


def find_best_result(ds_directory, filter):
    hpconfig_folders = find_subdirs(ds_directory)
    ds_name = os.path.split(ds_directory)[-1]
    test_metric = get_metric(ds_name, False)
    val_metric = get_metric(ds_name, True)
    # val_metric = test_metric
    val_selector, best_val = min_or_max(val_metric)
    best_result = None

    hparam_configs = cache.read_cached_json(ds_directory+'/hyperparams.json')

    for hpconfig_folder in hpconfig_folders:
        i = int(hpconfig_folder[9:])
        hpconfig = hparam_configs[i]
        if not filter(hpconfig):
            continue

        repeats_folder = os.path.join(ds_directory, hpconfig_folder)
        repeat_files = [entry.name for entry in os.scandir(
            repeats_folder) if entry.is_file() and entry.name.startswith('repeat_')]
        repeats_data = []

        for repeat_file in repeat_files:
            repeat_path = os.path.join(repeats_folder, repeat_file)
            repeat_data = cache.read_cached_json(repeat_path)
            repeats_data.append(repeat_data)

        if val_metric not in repeats_data[0]:
            print(i, repeats_data[0])
        avg_val = np.mean([rd[val_metric] for rd in repeats_data])

        if val_selector(avg_val, best_val) != best_val:
            # Standard deviation
            best_val = avg_val
            test_results = [rd[test_metric] for rd in repeats_data]
            avg_test = statistics.mean(test_results)
            std_test = statistics.stdev(test_results)
            best_result = {
                'val_metric': val_metric,
                'test_metric': test_metric,
                'hpconfig_folder': hpconfig_folder,
                'hpconfig': hpconfig,
                'avg_val': avg_val,
                'avg_test': avg_test,
                'std_test': std_test
            }

    return best_result


def filter_builder(**filter_params):
    def f(hparam_config):
        if hparam_config['regularization'] is None:
            if hparam_config['probability'] != 0.3:
                return False  # probability does not matter if no regularization is used => Fix 0.3 as a dummy value
            filter_params['probability'] = [0.3]
        for key, value_list in filter_params.items():
            if not isinstance(value_list, list):
                value_list = [value_list]
            if value_list is None or hparam_config[key] in value_list:
                continue
            return False
        return True
    return f


def create_grid_table(filters_params, datasets=datasets):
    param_combos = list(ParameterGrid(filters_params))
    columns = dict()
    for param in filters_params.keys():
        columns[to_lower_camel_case(param)] = []

    for param_combo in param_combos:
        for param, val in param_combo.items():
            columns[to_lower_camel_case(param)].append(",".join(map(str, val)))

    for dataset in datasets:
        avg_col = []
        std_col = []
        for param_combo in param_combos:
            filter = filter_builder(**param_combo)
            res = find_best_result(dataset, filter)
            avg_col.append(res["avg_test"])
            std_col.append(res["std_test"])
        ds_name = dataset.replace("ogbg-", "")
        columns[f"{ds_name}Avg"] = avg_col
        columns[f"{ds_name}Std"] = std_col

    return pd.DataFrame(columns)


def create_plot_table(filters_params, col_param, col_vals, dataset, no_row_tune=False):
    param_combos = list(ParameterGrid(filters_params))
    columns = dict()
    for param in filters_params.keys():
        columns[to_lower_camel_case(param)] = []

    for param_combo in param_combos:
        for param, val in param_combo.items():
            columns[to_lower_camel_case(param)].append(
                str(val))  # .append(",".join(map(str, val)))

    if no_row_tune:
        hpconfig = find_best_result(dataset, filter_builder(
            **{col_param: col_vals, **filters_params}))["hpconfig"]
    else:
        hpconfig = dict()

    for col_val in col_vals:
        avg_col = []
        std_col = []
        for param_combo in param_combos:
            filter = filter_builder(
                **{**hpconfig, col_param: [col_val], **param_combo})
            res = find_best_result(dataset, filter)
            avg_col.append(res["avg_test"])
            std_col.append(res["std_test"])

        columns[f"{to_lower_camel_case(col_val)}Avg"] = avg_col
        columns[f"{to_lower_camel_case(col_val)}Std"] = std_col

    return pd.DataFrame(columns)


def eval_table():
    """Creates the csv files."""
    grid: pd.DataFrame = create_grid_table({
        'convo_type': [['gcn'], ['gin']],
        'regularization': [[None], ['DropOut'], ['NodeSampling'], ['DropEdge'], ['GDC']]
    })
    grid.to_csv(f"{results_dir}/results.csv", sep=";", index_label="id")
    print(grid.shape)
    return grid


def eval_plot(model='gcn', axis='num_layers', dataset="ogbg-molfreesolv"):
    "Generates data for number of layers comparison for model + dataset"
    if axis == 'num_layers':
        params = {'num_layers': [1, 2, 3, 4, 5, 6]}
    elif axis == 'probability':
        params = {'probability': [0.3, 0.5, 0.7]}
    plot = create_plot_table({
        'convo_type': [model],
        **params
    }, 'regularization', [None, 'DropOut', 'NodeSampling', 'DropEdge', 'GDC'], dataset)
    plot.to_csv(
        f"{results_dir}/{axis}_plot_{dataset[5:]}_{model}.csv", sep=";", index_label="id")
    print(plot.shape)
    return plot


@click.command()
def cli():
    eval_table()
    eval_plot()


if __name__ == "__main__":
    cli()
