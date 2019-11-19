import os
import random
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
    nester = lambda xs: list(map(lambda t: tuple(map(lambda x: tf.constant(x, dtype=tf.float32), t)), zip(*xs)))
    x_train, x_test = nester(x_train), nester(x_test)
    return (x_train, y_train), (x_test, y_test)

def shuffle_dataset(x_train, y_train):
    indices = list(range(len(y_train)))
    random.shuffle(indices)
    return [x_train[index] for index in indices], [y_train[index] for index in indices]

def str_from_args(args):
    args_dict = vars(args)
    return ' '.join([key+'='+str(args_dict[key]) for key in sorted(args_dict.keys())])

def record_args(dataset_name, args, acc_avg, acc_std):
    experiments_filename = os.path.join(dataset_name+'_weights', 'experiments.txt')
    args_formatted = str_from_args(args)
    acc_formatted = (' acc=%.2f' % (acc_avg*100)) + (' std=%.2f' % (acc_std*100))
    with open(experiments_filename, 'a') as experiments_file:
        experiments_file.write(args_formatted + acc_formatted + '\n')
