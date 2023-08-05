import os
import csv
import json
import numpy as np

directory = ('/home/olga/GraphNeuralNetwork')
datasets = ['ogbg-molhiv', 'ogbg-molpcba',
            'ogbn-products', 'ogbn-proteins', 'ogbn-arxiv']

# # Get folders from the directory 
# def get_folders(directory):
#     folders = []
#     with os.scandir(directory) as entries:
#         for entry in entries:
#             if entry.is_dir():
#                 folders.append(entry.name)
#     return folders

# # Sort out the folders that are not of interest 
# def sort_out(dir_list): 

#     for item in dir_list: 
#         if item not in datasets: 
#             dir_list.delete(item)


# Get the folder paths --> use this method 
def get_folder_paths(directory, ds_list): 
    list_pf_paths = list()
    for entry in ds_list: 
        list_pf_paths.append(os.path.join(directory, entry))
    return list_pf_paths


def calculate_average_loss(ds_directory):
    hpconfig_folders = [entry.name for entry in os.scandir(
        ds_directory) if entry.is_dir()]
    best_result = None
    lowest_average_loss = float('inf')
    best_hpconfig = None

    for hpconfig_folder in hpconfig_folders:
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
            lowest_average_loss = average_loss
            best_hpconfig = hpconfig_folder

    # Create a CSV file to store the results
    results_file = os.path.join(ds_directory, 'results.csv')
    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['hpconfig_folder', 'hyperparams', 'average_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best_result)

    return results_file, best_hpconfig

def eval(dir_list): 
    for dir in dir_list:
        calculate_average_loss(dir)
        
