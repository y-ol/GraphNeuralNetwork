import numpy as np
import tensorflow as tf
import ogb
from ogb.graphproppred import GraphPropPredDataset
from ogb.nodeproppred import NodePropPredDataset
import funcy as fy
import math


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


def combine_graphs_labels(graph_batches, label_batches, include_mask=False):
    if not include_mask:
        yield from zip(graph_batches, label_batches)
    else:
        for graph_batch, label_batch in zip(graph_batches, label_batches):
            label_batch = np.array(label_batch)
            label_mask = label_batch == label_batch
            graph_batch["label_mask"] = label_mask
            yield graph_batch, label_batch[label_mask].reshape((-1, 1))


def make_tf_datasets(dataset, batchsize=30, include_mask=False):
    node_features = dataset.graphs[0]['node_feat'].shape[-1]
    split_idx = dataset.get_idx_split()
    final_batch = dict()
    final_labels = dict()
    for key, value in split_idx.items():
        final_labels[key] = list((fy.chunks(batchsize, dataset.labels[value])))

    for key, value in split_idx.items():
        final_batch[key] = list(map(batching, fy.chunks(
            batchsize, map(converter, np.array(dataset.graphs, dtype=object)[value]))))
    graph_signature = {'X': tf.TensorSpec(shape=(None, node_features), dtype=tf.float32),
                       'ref_A': tf.TensorSpec(shape=None, dtype=tf.int32),
                       'ref_B': tf.TensorSpec(shape=None, dtype=tf.int32),
                       'num_nodes': tf.TensorSpec(shape=None, dtype=tf.int32)}
    if include_mask:
        graph_signature['label_mask'] = tf.TensorSpec(
            shape=(None, dataset.num_tasks), dtype=tf.bool)
    output_signature = (graph_signature,
                        tf.TensorSpec(shape=(None, 1 if include_mask else dataset.num_tasks), dtype=tf.float32))
    for key, value in final_batch.items():
        final_batch[key] = tf.data.Dataset.from_generator(
            lambda: combine_graphs_labels(value, final_labels[key], include_mask), output_signature=output_signature)

    return final_batch
