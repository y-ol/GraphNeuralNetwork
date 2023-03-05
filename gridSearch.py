
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import json


search_space = {'num_layers': [1, 2, 3, 4, 5, 6],
                'learning_rate': [0.0001, 0.001, 0.01],
                'optimizer': ['Adam'],
                'regularization': ['DropOut', 'NodeSampling', 'DropEdge', 'GDC'],
                'probability': [0.3, 0.4, 0.5, 0.6],
                'activation_convo': ['ReLu', 'sigmoid', 'tanh'],
                'activation_dense': ['ReLu', 'sigmoid', 'tanh']}


# Experimental part


gnn_models = ['GCN', 'GIN']

# Create an empty dictionary to store the results
results = {}

# Loop over the search space and GNN models to run experiments
for gnn_model in gnn_models:
    for num_layers in search_space['num_layers']:
        for learning_rate in search_space['learning_rate']:
            for optimizer in search_space['optimizer']:
                for regularization in search_space['regularization']:
                    for probability in search_space['probability']:
                        for activation_convo in search_space['activation_convo']:
                            for activation_dense in search_space['activation_dense']:
                                # Create model with given hyperparameters
                                model = create_model(gnn_model, num_layers, learning_rate, optimizer,
                                                     regularization, probability, activation_convo, activation_dense)

                                # Train and evaluate the model
                                accuracy = train_and_evaluate(model)

                                # Add results to dictionary
                                results[(gnn_model, num_layers, learning_rate, optimizer, regularization,
                                         probability, activation_convo, activation_dense)] = accuracy

# Save results to a JSON file
with open('results.json', 'w') as f:
    json.dump(results, f)
