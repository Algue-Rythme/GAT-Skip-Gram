import os
from collections import defaultdict
import numpy as np
import tensorflow as tf
import numpy as np


def standardize_features(features):
    mean = tf.math.reduce_mean(features, axis=0, keepdims=True)
    std = tf.math.reduce_std(features, axis=0, keepdims=True)
    features = (features - mean) / std
    return features

def get_features(features_file, standardize):
    with open(features_file, 'r') as f:
        tokens = f.readline()
        features = []
        while tokens:
            line = list(map(float, tokens.split(',')))
            features.append(line)
            tokens = f.readline()
    features = tf.constant(features, dtype=tf.float32)
    if standardize:
        features = standardize_features(features)
    return features.numpy()

def print_statistics(graph_adj, graph_features):
    num_nodes = [int(graph.shape[0]) for graph in graph_adj]
    num_edges = [int(tf.reduce_sum(graph)/2.) for graph in graph_adj]
    print('num_graphs: %d'%len(graph_adj))
    print('num_nodes: %d'%sum(num_nodes))
    print('num_edges: %d'%sum(num_edges))
    print('avg_nodes: %.2f'%np.array(num_nodes).mean())
    print('avg_edges: %.2f'%np.array(num_edges).mean())
    print('num_features: %d'%int(graph_features[0].shape[1]))

def read_dortmund(prefix, standardize):
    print('opening %s...'%prefix, flush=True)
    graph_file = os.path.join(prefix, '%s_graph_indicator.txt'%prefix)
    graph_ids = []
    graph_nodes = defaultdict(list)
    new_node_ids = dict()
    with open(graph_file, 'r') as f:
        tokens = f.readline()
        node_id = 0
        while tokens:
            graph_id = int(tokens)-1
            graph_ids.append(graph_id)
            new_node_ids[node_id] = len(graph_nodes[graph_id])
            graph_nodes[graph_id].append(node_id)
            node_id += 1
            tokens = f.readline()
    graph_adj = [np.zeros(shape=(len(nodes),len(nodes)), dtype=np.float32) for _, nodes in graph_nodes.items()]
    adj_file = os.path.join(prefix, '%s_A.txt'%prefix)
    with open(adj_file, 'r') as f:
        tokens = f.readline()
        while tokens:
            tokens = tokens.split(',')
            node_a, node_b = int(tokens[0])-1, int(tokens[1])-1
            graph_a, graph_b = graph_ids[node_a], graph_ids[node_b]
            assert graph_a == graph_b
            node_a, node_b = new_node_ids[node_a], new_node_ids[node_b]
            graph_adj[graph_a][node_a, node_b] = 1.
            tokens = f.readline()
    graph_adj = [tf.constant(adj, dtype=tf.float32) for adj in graph_adj]
    features_file = os.path.join(prefix, '%s_node_attributes.txt'%prefix)
    features = get_features(features_file, standardize)
    num_features = int(features.shape[1])
    graph_features = [np.zeros(shape=(len(nodes), num_features), dtype=np.float32) for nodes in graph_adj]
    for node_id, graph_id in enumerate(graph_ids):
        new_node_id = new_node_ids[node_id]
        graph_features[graph_id][new_node_id,:] = features[node_id,:]
    graph_features = [tf.constant(graph, dtype=tf.float32) for graph in graph_features]
    print('%s opened with success !'%prefix, flush=True)
    print_statistics(graph_adj, graph_features)
    return graph_adj, graph_features
