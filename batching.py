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


def make_inputs(dataset):

    split_idx = dataset.get_idx_split()
    final_batch = dict()
    final_labels = dict()
    for key, value in split_idx.items():
        final_labels[key] = list((fy.chunks(30, dataset.labels[value])))

    for key, value in split_idx.items():
        final_batch[key] = list(map(batching, fy.chunks(
            30, map(converter, np.array(dataset.graphs, dtype=object)[value]))))
    for key, value in final_batch.items():
        final_batch[key] = (tf.data.Dataset.from_generator((lambda: zip(value, final_labels[key])), output_signature=(({'X': tf.TensorSpec(shape=(None, 9), dtype=tf.float32),
                                                                                                                        'ref_A': tf.TensorSpec(shape=None, dtype=tf.int32),
                                                                                                                        'ref_B': tf.TensorSpec(shape=None, dtype=tf.int32),
                                                                                                                        'num_nodes': tf.TensorSpec(shape=None, dtype=tf.int32)},
                                                                                                                       tf.TensorSpec(shape=None, dtype=tf.float32)))))

    return final_batch
