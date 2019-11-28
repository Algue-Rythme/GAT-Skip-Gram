import datetime
import os
import random
import warnings
import numpy as np
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
