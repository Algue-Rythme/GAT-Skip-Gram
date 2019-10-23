import os
from collections import defaultdict
import tensorflow as tf
import numpy as np


def standardize_features(features):
    mean = tf.math.reduce_mean(features, axis=0, keepdims=True)
    std = tf.math.reduce_std(features, axis=0, keepdims=True)
    features = (features - mean) / std
    return features

def get_features(features_file, standardize):
    with open(features_file, 'r') as f:
        tokens = f.readline().split(',')
        features = []
        while tokens:
            line = map(float, tokens)
            features.append(line)
            tokens = f.readline().split(',')
    features = tf.constant(features, dtype=tf.float32)
    if standardize:
        features = standardize_features(features)
    return features

def read_dortmund(prefix, standardize):
    graph_file = os.path.join(prefix, 'DS_graph_indicator.txt')
    graph_ids = []
    graph_nodes = defaultdict(list)
    new_node_ids = dict()
    with open(graph_file, 'r') as f:
        tokens = f.readline()
        node_id = 0
        while tokens:
            graph_id = int(tokens)
            graph_ids.append(graph_id)
            new_node_ids[node_id] = len(graph_nodes[graph_id])
            graph_nodes[graph_id].append(node_id)
            node_id += 1
            tokens = f.readline()
    graph_adj = [np.zeros(shape=(len(nodes),len(nodes)), dtype=np.float32) for _, nodes in graph_nodes]
    adj_file = os.path.join(prefix, 'DS_A.txt')
    with open(adj_file, 'r') as f:
        tokens = f.readline().split()
        while tokens:
            node_a, node_b = tokens
            graph_a, graph_b = graph_ids[node_a], graph_ids[node_b]
            assert graph_a == graph_b
            node_a, node_b = new_node_ids[node_a], new_node_ids[node_b]
            graph_adj[graph_a][node_a, node_b] = 1.
            tokens = f.readline().split()
    graph_adj = [tf.constant(adj, dtype=tf.float32) for adj in graph_adj]
    features_file = os.path.join(prefix, 'DS_node_attributes.txt')
    features = get_features(features_file, standardize)
    num_features = int(features.shape[1])
    graph_features = [tf.constant(len(nodes), num_features) for nodes in range(graph_adj)]
    for node_id, graph_id in enumerate(graph_ids):
        new_node_id = new_node_ids[node_id]
        graph_features[graph_id][new_node_id,:] = features[node_id,:]
    return graph_adj, graph_features
