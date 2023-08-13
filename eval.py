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



def calculate_average_loss(ds_directory, filter):
    # TODO min, max (vor mse min, otherwise max )
    # TODO Standard deviation --done
    # TODO different metrics as return -- lookup table done 
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
        losses = []

        for repeat_file in repeat_files:
            repeat_path = os.path.join(repeats_folder, repeat_file)
            with open(repeat_path, 'r') as f:
                repeat_data = json.load(f)
                loss = repeat_data['val_loss']
                losses.append(loss)

        average_loss = sum(losses) / len(losses)
        
        if average_loss < lowest_average_loss:
            best_result=({'hpconfig_folder': hpconfig_folder,'hyperparams': repeat_data['hyperparams'],'average_loss': average_loss})
            # Standard daviation 
            deviation_list = []
            for repeat_file in repeat_files: 
                with open(os.path.join(repeats_folder, repeat_file), 'r') as f: 
                    deviation_list.append(json.load(f)['val_loss'])

            lowest_average_loss = average_loss
            best_hpconfig = hpconfig
            std_dev = statistics.stdev(deviation_list)
            # dictionary with all the other values
    res_2 = dict()
    res_2['best_hpconfig'] = hpconfig_folder
    res_2['average_loss'] = average_loss
    res_2['std_dev'] = std_dev

    return best_hpconfig, res_2


def eval(hpconfig): 
    """Creates the csv file. Takes in dataset-hpconfig-best and writes the csv file in the"""


# lookup string metric in the json file
def lookup(dataset):
    if dataset == 'ogbg-molhiv':
        metric = tf.keras.metrics.AUC(curve='ROC')
    elif dataset == 'ogbg-molpcba':
        metric = tf.keras.metrics.AUC(curve='PR')
    elif (dataset == 'ogbg-molesol' or dataset == 'ogbg-molfreesolv' or dataset == 'ogbg-mollipo'):
        metric = tf.keras.metrics.MeanSquaredError()
    else:
        raise Exception("dataset unknown")

    return metric


def min_or_max(metric): 
    if(metric == 'mse'): 
        return 'min'
    else: 
        return 'max'

def filter(convo_type, regularisation): 
    def f(hparam_config):
        if convo_type == hparam_config['convo_type'] and regularisation == hparam_config['regularization']: 
            return True
        else: 
            return False
    return f 


# TODO Create a CSV file to store the results
def create_csv(ds_directory, best_result): 
    results_file = os.path.join(ds_directory, 'results.csv')
    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['hpconfig_folder', 'hyperparams', 'average_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best_result)
