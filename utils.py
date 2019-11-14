import random
import matplotlib as plt
import numpy as np
import tensorflow as tf
try:
    import pygsp
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
