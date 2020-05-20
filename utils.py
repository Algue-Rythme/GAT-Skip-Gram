import datetime
import functools
import os
import random
import warnings
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
try:
    import pygsp
    import matplotlib as plt
    def plot_pyramid(A_pyramid, kind='spring'):
        graph = pygsp.graphs.Graph(A_pyramid[0])
        graph.set_coordinates(kind)
        coords = graph.coords
        for adj, indices in A_pyramid[1:]:
            signal = np.zeros(graph.N)
            signal[indices] = 1.
            pygsp.plotting.plot_signal(graph, signal)
            graph = pygsp.graphs.Graph(adj)
            coords = coords[indices,:]
            graph.set_coordinates(coords)
        pygsp.plotting.plot_graph(graph)
        plt.pyplot.show()
except ImportError:
    warnings.warn('Unsupported pygsp or matplotlib')
    def plot_pyramid(_A_pyramid, _kind='spring'):
        pass


def train_test_split(*lsts, **kwargs):
    len_x = len(lsts[0])
    assert all([len_x == len(x) for x in lsts])
    indices = tf.range(start=0, limit=len_x, dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    shuffled_lsts = [[x[index] for index in shuffled_indices] for x in lsts]
    train_lim = int(kwargs['split_ratio'] * len_x)
    train = [x[train_lim:] for x in shuffled_lsts]
    test = [x[:train_lim] for x in shuffled_lsts]
    x_train, y_train = train[:-1], train[-1]
    x_test, y_test = test[:-1], test[-1]
    return (x_train, y_train), (x_test, y_test)

def nester(xs):
    def make_float32_tensor(x):
        return tf.constant(x, dtype=tf.float32)
    def tuple_tensor_from_tuple(t):
        return tuple(map(make_float32_tensor, t))
    return list(map(tuple_tensor_from_tuple, zip(*xs)))

def shuffle_dataset(x_train, y_train):
    indices = list(range(len(y_train)))
    random.shuffle(indices)
    return [x_train[index] for index in indices], [y_train[index] for index in indices]

def str_from_args(args):
    args_dict = vars(args)
    return ' '.join([key+'='+str(args_dict[key]) for key in sorted(args_dict.keys())])

def get_now():
    return '%s-%s'%(datetime.date.today().strftime('%m/%d/%y'), datetime.datetime.now())

def record_args(method_name, departure_time, dataset_name, args, acc_avg, acc_std):
    experiments_filename = os.path.join(dataset_name+'_weights', 'experiments.txt')
    args_formatted = str_from_args(args)
    acc_formatted = (' acc=%.2f'%(acc_avg*100)) + (' std=%.2f'%(acc_std*100))
    departure_time = ' start=' + departure_time
    end_time = ' end=' + get_now()
    with open(experiments_filename, 'a') as experiments_file:
        experiments_file.write(args_formatted + acc_formatted + ' method_name=%s'%method_name)
        experiments_file.write(departure_time + end_time + '\n')

def build_laplacian(adj):
    degrees = tf.linalg.diag(tf.math.reduce_sum(adj, axis=0))
    laplacian = degrees - adj
    return laplacian

def get_degrees(A):
    return tf.math.reduce_sum(A, axis=1)

def normalize_adjacency(A, rooted_subtree, identity=True):
    if identity:
        A = A + tf.eye(num_rows=A.shape[0])
    D = get_degrees(A)
    D = tf.linalg.diag(tf.math.rsqrt(D))
    A = D @ A @ D
    if rooted_subtree:
        A = tf.linalg.set_diag(A, tf.zeros(shape=int(A.shape[0]), dtype=tf.float32))
    return A

def dispatch(graph_inputs, index):
    return [graph_input[index] for graph_input in graph_inputs]

def get_weight_filenames(dataset_name):
    try:
        os.mkdir(dataset_name+'_weights')
    except FileExistsError:
        pass
    finally:
        wl_embedder_file = os.path.join(dataset_name+'_weights', 'wl_embedder.h5')
        graph_embedder_file = os.path.join(dataset_name+'_weights', 'graph_embedder.h5')
        csv_file = os.path.join(dataset_name+'_weights', 'graph_embeddings.csv')
    return wl_embedder_file, graph_embedder_file, csv_file

def memoize(func):
    cache = func.cache = {}
    @functools.wraps(func)
    def memoized_func(*l_args):
        key = tuple([id(arg) for arg in l_args])
        if key not in cache:
            cache[key] = func(*l_args)
        return cache[key]
    return memoized_func

def get_graph_labels(dataset_name):
    labels_filename = os.path.join(dataset_name, '%s_graph_labels.txt'%dataset_name)
    with open(labels_filename, 'r') as f:
        labels_data = np.loadtxt(f, ndmin=1).astype(dtype=np.int64)
    label_encoder = LabelEncoder()
    label_encoder.fit(labels_data)
    return label_encoder.transform(labels_data)

def get_data(dataset_name, graph2vec):
    if graph2vec:
        graph_embedder_filename = os.path.join('../graph2vec/features/', dataset_name+'.csv')
    else:
        graph_embedder_filename = os.path.join(dataset_name+'_weights', 'graph_embeddings.csv')
    with open(graph_embedder_filename, 'r') as f:
        if graph2vec:
            embeddings_data = np.loadtxt(f, delimiter=',', skiprows=1).astype(np.float32)
            embeddings_data = embeddings_data[:,1:]  # remove node index
        else:
            embeddings_data = np.loadtxt(f, delimiter='\t').astype(np.float32)
    labels_data = get_graph_labels(dataset_name)
    return embeddings_data, labels_data

def print_split_indexes_to_csv(dataset_name, train_indexes, test_indexes):
    filename = os.path.join(dataset_name+'_weights', 'graph_indexes.csv')
    with open(filename, 'w') as file:
        file.write('\t'.join(map(str, train_indexes))+'\n')
        file.write('\t'.join(map(str, test_indexes))+'\n')

def get_train_test_indexes(dataset_name):
    filename = os.path.join(dataset_name+'_weights', 'graph_indexes.csv')
    with open(filename, 'r') as file:
        train_indexes = list(map(int, file.readline().split()))
        test_indexes = list(map(int, file.readline().split()))
    return train_indexes, test_indexes
