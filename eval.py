import tensorflow as tf
import os
import csv
import json
import numpy as np
import statistics

directory = ('/home/olga/GraphNeuralNetwork')
datasets = ['ogbg-molhiv', 'ogbg-molpcba',
            'ogbn-products', 'ogbn-proteins', 'ogbn-arxiv']


# Get the folder paths --> use this method 
def get_folder_paths(directory, ds_list): 
    list_pf_paths = list()
    for entry in ds_list: 
        list_pf_paths.append(os.path.join(directory, entry))
    return list_pf_paths

def get_metric(dataset_name):
    if dataset_name == 'ogbg-molhiv': 
        return 'rocauc'
    elif dataset_name == 'ogbg-molpcba': 
        return 'ap'
    return 'loss'

def calculate_average_loss(ds_directory, filter):
    hpconfig_folders = [entry.name for entry in os.scandir(
        ds_directory) if entry.is_dir()]
    best_result = None
    lowest_average_loss = float('inf')
    best_hpconfig = None
    average_loss = None 
    std_dev = None

    with open(ds_directory+'/hyperparams.json', 'r') as f: 
        hparam_configs = json.load(f)
        
    for hpconfig_folder in hpconfig_folders:
        i = int(hpconfig_folder[9:])
        hpconfig = hparam_configs[i]
        if not filter(hpconfig):
            continue 
            
        repeats_folder = os.path.join(ds_directory, hpconfig_folder)
        repeat_files = [entry.name for entry in os.scandir(
            repeats_folder) if entry.is_file() and entry.name.startswith('repeat_')]
        val_losses = []

        for repeat_file in repeat_files:
            repeat_path = os.path.join(repeats_folder, repeat_file)
            with open(repeat_path, 'r') as f:
                repeat_data = json.load(f)
                val_loss = repeat_data['val_loss']
                val_losses.append(val_loss)

        average_loss = sum(val_losses) / len(val_losses)
        
        if average_loss < lowest_average_loss:
            best_result=({'hpconfig_folder': hpconfig_folder,'hyperparams': repeat_data['hyperparams'],'average_loss': average_loss})
            # Standard deviation 
            test_metric_list = []
            for repeat_file in repeat_files: 
                with open(os.path.join(repeats_folder, repeat_file), 'r') as f: 
                    test_metric_list.append(json.load(
                        f)['test_' + get_metric(os.path.split(ds_directory)[-1])])

            lowest_average_loss = average_loss
            best_hpconfig = hpconfig
            std_dev = statistics.stdev(test_metric_list)
            mean = statistics.mean(test_metric_list)
            # dictionary with all the other values
    res_2 = dict()
    res_2['best_hpconfig'] = hpconfig_folder
    res_2['average_loss'] = average_loss
    res_2['std_dev'] = std_dev
    res_2['mean'] = mean

    return best_hpconfig, res_2


def eval(hpconfig): 
    """Creates the csv file. Takes in dataset-hpconfig-best and writes the csv file in the"""



# lookup string metric in the json file
def lookup(dataset):
    if dataset == 'ogbg-molhiv':
        metric = 'test_rocauc'
    elif dataset == 'ogbg-molpcba':
        metric = 'test_ap'
    elif (dataset == 'ogbg-molesol' or dataset == 'ogbg-molfreesolv' or dataset == 'ogbg-mollipo'):
        metric = 'test_mse' # assuming it's called mse in the keras evaluator  
    else:
        raise Exception("dataset unknown")

    return metric


def min_or_max(metric): 
    if(metric == 'mse'): 
        return 'min'
    else: 
        return 'max'

def filter_builder(**filter_params): 
    def f(hparam_config):
        for key, value_list in filter_params.items():
            if value_list is None or hparam_config[key] in value_list:
                continue 
            return False
        return True 
    return f 



