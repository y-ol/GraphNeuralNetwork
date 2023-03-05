
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
