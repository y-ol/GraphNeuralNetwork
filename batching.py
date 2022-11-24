import numpy as np
import tensorflow as tf
import ogb
from ogb.graphproppred import GraphPropPredDataset
import funcy as fy


def batching(graphs):
    X = []
    ref_A = []
    ref_B = []
    i = 0
    num_nodes = []
    for graph in graphs:
        X.append(graph['X'])
        ref_A.append(graph['ref_A']+i)
        ref_B.append(graph['ref_B']+i)
        i += len(graph['X'])
        num_nodes.append(len(graph['X']))
    return {'X': np.concatenate(X, axis=0),
            'ref_A': np.concatenate(ref_A),
            'ref_B': np.concatenate(ref_B),
            'num_nodes': np.array(num_nodes)}


def converter(graph):
    return {'X': graph['node_feat'],
            'ref_A': graph['edge_index'][0],
            'ref_B': graph['edge_index'][1]
            }


dataset = GraphPropPredDataset(name='ogbg-molhiv')

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
labels = dataset.labels
i = 0
graph, label = dataset[i]  # destructuring
num_Graphs = len(dataset)

ref_A = dataset.graphs[0]["edge_index"][0]
ref_B = dataset.graphs[0]["edge_index"][1]

# fy.first(fy.chunks(30, dataset.graphs))  # OGB --- not saved in any var

# returns an iterable object of type map - not subscriptable
chunks = map(batching, fy.chunks(30, map(converter, dataset.graphs)))
first_batch = fy.first(chunks)  # type = class dictionary


""""do: 
take chunks and create 30x 'X', 30x 'ref_A, 30x 'ref_B' 
counter for list 

"""

# input = chunks -> partition of whole  dataset in 30


def create_batches(dataset):
    list_of_batches = tf.constant([])
    i = 0
    for entry in dataset.graphs:
        list_of_batches[i] = fy.first(fy.chunks)
        i += len(chunks[i]['X'] + chunks[i]['ref_A'] + chunks[i]['ref_B'])
    return list_of_batches
