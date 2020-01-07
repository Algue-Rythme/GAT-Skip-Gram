import itertools
from itertools import chain
from itertools import product
import os
from collections import defaultdict
import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds


def standardize_features(features):
    mean = tf.math.reduce_mean(features, axis=0, keepdims=True)
    std = tf.math.reduce_std(features, axis=0, keepdims=True)
    features = (features - mean) / std
    return features

def node_features_from_attribute_file(prefix, standardize):
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

def read_label_file(label_file):
    with open(label_file, 'r') as f:
        tokens = f.readline()
        labels = []
        label_set = set()
        while tokens:
            label = int(tokens)
            label_set.add(label)
            labels.append(label)
            tokens = f.readline()
    return labels, label_set

def node_features_from_label_file(prefix):
    label_file = os.path.join(prefix, '%s_node_labels.txt'%prefix)
    try:
        labels, label_set = read_label_file(label_file)
    except IOError:
        return None
    labels = tf.keras.backend.one_hot(labels, len(label_set))
    return tf.constant(labels, dtype=tf.float32)

def get_node_features(prefix, standardize, graph_ids, new_node_ids, graph_adj):
    node_attributes = node_features_from_attribute_file(prefix, standardize)
    labels = node_features_from_label_file(prefix)
    node_features = node_attributes if node_attributes is not None else labels
    if node_attributes is not None and labels is not None:
        node_features = tf.concat([node_features, labels], axis=1)
    if node_attributes is None and labels is None:
        node_features = [tf.math.reduce_sum(adj, axis=-1, keepdims=True) for adj in graph_adj]
        return node_features
    del node_attributes, labels
    num_node_features = int(node_features.shape[1])
    graph_node_features = [np.zeros(shape=(len(nodes), num_node_features), dtype=np.float32) for nodes in graph_adj]
    for node_id, graph_id in enumerate(graph_ids):
        new_node_id = new_node_ids[node_id]
        graph_node_features[graph_id][new_node_id,:] = node_features[node_id,:]
    graph_node_features = [tf.constant(graph, dtype=tf.float32) for graph in graph_node_features]
    return graph_node_features

def get_graph_node_ids(prefix):
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
    return graph_ids, graph_nodes, new_node_ids

def get_graph_adj(prefix, graph_nodes, graph_ids, new_node_ids):
    graph_adj = [np.zeros(shape=(len(nodes),len(nodes)), dtype=np.float32) for _, nodes in graph_nodes.items()]
    adj_file = os.path.join(prefix, '%s_A.txt'%prefix)
    adj_lst = []
    with open(adj_file, 'r') as f:
        tokens = f.readline()
        while tokens:
            tokens = tokens.split(',')
            node_a, node_b = int(tokens[0])-1, int(tokens[1])-1
            adj_lst.append((node_a, node_b))
            graph_a, graph_b = graph_ids[node_a], graph_ids[node_b]
            assert graph_a == graph_b
            node_a, node_b = new_node_ids[node_a], new_node_ids[node_b]
            graph_adj[graph_a][node_a, node_b] = 1.
            tokens = f.readline()
    # graph_adj = [tf.constant(adj, dtype=tf.float32) for adj in graph_adj]
    return graph_adj, adj_lst

def get_edge_features(prefix, _, graph_nodes, graph_ids, new_node_ids, adj_lst):
    label_file = os.path.join(prefix, '%s_edge_labels.txt'%prefix)
    try:
        labels, label_set = read_label_file(label_file)
    except IOError:
        return None
    graph_edge_labels = [np.zeros(shape=(len(nodes),len(nodes)), dtype=np.int32) for _, nodes in graph_nodes.items()]
    for label, (node_a, node_b) in zip(labels, adj_lst):
        graph_a = graph_ids[node_a]
        new_node_a, new_node_b = new_node_ids[node_a], new_node_ids[node_b]
        graph_edge_labels[graph_a][new_node_a, new_node_b] = label
    graph_edge_labels = [tf.keras.backend.one_hot(graph, len(label_set)) for graph in graph_edge_labels]
    return graph_edge_labels

def print_statistics(graph_adj, node_features, edge_features=None):
    num_nodes = [int(graph.shape[0]) for graph in graph_adj]
    num_edges = [int(tf.reduce_sum(graph)/2.) for graph in graph_adj]
    print('num_graphs: %d'%len(graph_adj))
    print('num_nodes: %d'%sum(num_nodes))
    print('num_edges: %d'%sum(num_edges))
    print('avg_nodes: %.2f'%np.array(num_nodes).mean())
    print('avg_edges: %.2f'%np.array(num_edges).mean())
    print('num_node_features: %d'%int(node_features[0].shape[1]))
    if edge_features is not None:
        print('num_edge_features: %d'%int(edge_features[0].shape[2]))
    else:
        print('no edge features')

def read_dortmund(prefix, with_edge_features, standardize):
    graph_ids, graph_nodes, new_node_ids = get_graph_node_ids(prefix)
    graph_adj, adj_lst = get_graph_adj(prefix, graph_nodes, graph_ids, new_node_ids)
    node_features = get_node_features(prefix, standardize, graph_ids, new_node_ids, graph_adj)
    edge_features = None
    if with_edge_features:
        edge_features = get_edge_features(prefix, standardize, graph_nodes, graph_ids, new_node_ids, adj_lst)
    if edge_features is None:
        return node_features, graph_adj
    return node_features, graph_adj, edge_features

def read_dataset(name, with_edge_features, standardize):
    print('opening %s...'%name, flush=True)
    graph_inputs = read_dortmund(name, with_edge_features, standardize)
    print('%s opened with success !'%name, flush=True)
    print_statistics(*graph_inputs)
    return graph_inputs

def read_graph_labels(dataset_name):
    labels_filename = os.path.join(dataset_name, '%s_graph_labels.txt'%dataset_name)
    labels, label_set = read_label_file(labels_filename)
    class_indexes_remapper = dict(zip(label_set, range(len(label_set))))
    labels = [class_indexes_remapper[label] for label in labels]
    return labels, len(label_set)

def read_image(image):
    image = tf.squeeze(image) if len(image.shape) == 2 else tf.math.reduce_mean(image, axis=-1)
    mask = tf.not_equal(image, tf.zeros(shape=image.shape, dtype=image.dtype))
    indices = tf.where(mask)
    luminosity = tf.gather_nd(image, indices)
    luminosity = tf.dtypes.cast(luminosity, dtype=tf.float32) / tf.constant(255., dtype=tf.float32)
    luminosity = tf.expand_dims(luminosity, axis=-1)
    indices_features = tf.dtypes.cast(indices, dtype=tf.float32)
    features = tf.concat([indices_features, luminosity], axis=1)
    features = features.numpy()
    indices = indices.numpy()
    indices_right = indices + np.array([0, 1], dtype=indices.dtype)
    indices_down = indices + np.array([1, 0], dtype=indices.dtype)
    iter_indices_right = product(enumerate(indices), enumerate(indices_right))
    iter_indices_down = product(enumerate(indices), enumerate(indices_down))
    adj_lst = []
    for (i, idi), (j, idj) in chain(iter_indices_right, iter_indices_down):
        if np.array_equal(idi, idj):
            adj_lst.append([i, j])
            adj_lst.append([j, i])
    return adj_lst, features

def init_data(data_type, parts):
    data = tfds.load(data_type, with_info=False)
    if parts == 'all':
        data = itertools.chain(data['train'], data['test'])
        num_data = 70 * 1000
    elif part in ['train', 'test']:
        data = data[part]
        num_data = 10 * 1000 if part == 'test' else 60 * 1000
    else:
        raise ValueError
    progbar = tf.keras.utils.Progbar(num_data,
                                     stateful_metrics=['num_nodes', 'num_edges'])
    prefix = data_type.upper() + '_' + part
    try:
        os.mkdir(prefix)
    except FileExistsError:
        pass
    return data, progbar, prefix

def produce_data_images(data_type, parts):
    data, progbar, prefix = init_data(data_type, parts)
    map_name = lambda name: os.path.join(prefix, name)
    with open(map_name(prefix+'_graph_indicator.txt'), 'w') as indicator_file, \
         open(map_name(prefix+'_A.txt'), 'w')  as adj_file, \
         open(map_name(prefix+'_node_attributes.txt'), 'w') as features_file:
        num_nodes, num_edges = 1, 0
        for step, image_label in enumerate(data):
            adj, features = read_image(image_label['image'])
            for a, b in adj:
                adj_file.write(str(a+num_nodes)+', '+str(b+num_nodes)+'\n')
            for feature in features:
                indicator_file.write(str(step+1)+'\n')
                features_file.write(', '.join(map(str, feature.tolist()))+'\n')
            num_nodes += int(features.shape[0])
            num_edges += len(adj)
            progbar.update(step+1, [('num_nodes', num_nodes+1), ('num_edges', num_edges)])

def produce_data_labels(data_type, parts):
    data, progbar, prefix = init_data(data_type, parts)
    with open(os.path.join(prefix, prefix+'_graph_labels.txt'), 'w') as labels_file:
        for step, image_label in enumerate(data):
            label = int(image_label['label'].numpy())
            labels_file.write(str(label)+'\n')
            progbar.update(step+1)

def available_tasks():
    tasks = ['ENZYMES', 'PROTEINS', 'PROTEINS_full', 'MUTAG',
             'PTC_FM', 'NCI1', 'PTC_FR', 'DD',
             # 'Letter-high', 'Letter-med', 'Letter-low',
             # 'REDDIT_BINARY', 'COLLAB', 'MCF-7', 'MCF-7H',
             'DLA', 'IMDB-BINARY', 'MNIST_test',
             'FASHION_MNIST_test', 'CIFAR10_test']
    return tasks


if __name__ == '__main__':
    with tf.device('/cpu'):
        confirm = input('Are you sure to generate ? This is long. Tape "yes" or exit. ')
        if confirm == 'yes':
            data_name = input('Name of the dataset: \'mnist\', \'fashion_mnist\' or \'cifar10\'. ')
            part = input('Choose between: train, test, all. ')
            category = input('What do you want to generate ? labels or graphs. ')
            if category == 'graphs':
                produce_data_images(data_name, part)
            elif category == 'labels':
                produce_data_labels(data_name, part)
            else:
                raise ValueError
        else:
            print('Exit procedure.')