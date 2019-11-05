import os
from collections import defaultdict
import numpy as np
import tensorflow as tf


def standardize_features(features):
    mean = tf.math.reduce_mean(features, axis=0, keepdims=True)
    std = tf.math.reduce_std(features, axis=0, keepdims=True)
    features = (features - mean) / std
    return features

def features_from_attribute_file(prefix, standardize):
    attribute_file = os.path.join(prefix, '%s_node_attributes.txt'%prefix)
    try:
        with open(attribute_file, 'r') as f:
            tokens = f.readline()
            attributes = []
            while tokens:
                line = list(map(float, tokens.split(',')))
                attributes.append(line)
                tokens = f.readline()
        if standardize:
            attributes = tf.constant(attributes, dtype=tf.float32)
            attributes = standardize_features(attributes)
        else:
            attributes = np.array(attributes, dtype=np.float32)
        return attributes
    except IOError:
        return None

def features_from_label_file(prefix):
    label_file = os.path.join(prefix, '%s_node_labels.txt'%prefix)
    try:
        with open(label_file, 'r') as f:
            tokens = f.readline()
            labels = []
            label_set = set()
            while tokens:
                label = int(tokens)
                label_set.add(label)
                labels.append(label)
                tokens = f.readline()
        labels = tf.keras.backend.one_hot(labels, len(label_set))
        return tf.constant(labels, dtype=tf.float32)
    except IOError:
        return None

def get_features(prefix, standardize):
    attributes = features_from_attribute_file(prefix, standardize)
    labels = features_from_label_file(prefix)
    features = attributes if attributes is not None else labels
    if attributes is not None and labels is not None:
        features = tf.concat([features, labels], axis=1)
    return features

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
    features = get_features(prefix, standardize)
    num_features = int(features.shape[1])
    graph_features = [np.zeros(shape=(len(nodes), num_features), dtype=np.float32) for nodes in graph_adj]
    for node_id, graph_id in enumerate(graph_ids):
        new_node_id = new_node_ids[node_id]
        graph_features[graph_id][new_node_id,:] = features[node_id,:]
    graph_features = [tf.constant(graph, dtype=tf.float32) for graph in graph_features]
    print('%s opened with success !'%prefix, flush=True)
    print_statistics(graph_adj, graph_features)
    return graph_adj, graph_features
