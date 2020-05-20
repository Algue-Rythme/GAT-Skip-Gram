import argparse
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pygsp
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
import dataset
import loukas
import utils

def retrieve_coordinates(features):
    assert int(features.shape[1]) == 3
    coords = np.array(features[:,0:2])  # first two are coordinates
    x = coords[:,1]
    y = -coords[:,0]  # other direction
    coords = np.stack([x, y], axis=-1)
    coords = coords * 28
    return coords

def print_graphs(Calls, Galls, attributes=None):
    size = 3
    alpha=0.55
    edge_width=0.8
    node_size=20
    fig = plt.figure(figsize=(len(Galls)*size*3, size))
    zipped = zip(Calls, Galls, [None]*len(Calls) if attributes is None else attributes)
    for i, (Call, Gall, features) in enumerate(zipped):
        G = Gall[0]
        C = Call[0]
        C = C.toarray()
        # print(G, C)
        if features is None:
            G.set_coordinates(kind='spring')
        else:
            coords = retrieve_coordinates(features)
            G.set_coordinates(coords)
        # pygsp.plotting.plot_graph(G)
        ax = fig.add_subplot(1, len(Galls), i+1)
        ax.axis('off')
        [x,y] = G.coords.T
        edges = np.array(G.get_edge_list()[0:2])
        for eIdx in range(0, edges.shape[1]):
            ax.plot(x[edges[:,eIdx]], y[edges[:,eIdx]], color='k', alpha=alpha, lineWidth=edge_width)
        for j in range(int(C.shape[0])):
            subgraph = np.arange(G.N)[C[j,:] > 0]
            ax.scatter(x[subgraph], y[subgraph], c='b', s=node_size, alpha=alpha)


def get_distances_indexes(dataset_name, n_neighbors, load_neighbors):
    X, _ = utils.get_data(dataset_name, graph2vec=False)
    fname = os.path.join(dataset_name+'_weights', 'neighbors.csv')
    if not load_neighbors:
        print('Compute Nearest Neighbors')
        nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
        distances, indices = nn.kneighbors(X)
        del X
        with open(fname, 'w') as file:
            for dst, indexes in zip(distances, indices):
                numbers = map(str, dst.tolist() + indexes.tolist())
                msg = ','.join(numbers)
                file.write(msg+'\n')
    else:
        frame = np.loadtxt(fname, delimiter=',')
        distances = frame[:,:n_neighbors]
        indices = frame[:,n_neighbors:].astype(np.int64)
    return distances, indices

def find_nearest_neighbor(dataset_name, n_neighbors, load_neighbors, reverse, print_pyramid=True):
    distances, indices = get_distances_indexes(dataset_name, n_neighbors, load_neighbors)
    all_graphs = dataset.read_dataset(args.task,
                                      with_edge_features=False,
                                      standardize=True)
    FEATURES_IDX, ADJ_IDX = 0, 1
    graphs_features = all_graphs[FEATURES_IDX]
    graphs_adj = all_graphs[ADJ_IDX]
    k = 18
    r = 0.99
    zipped = list(zip(distances, indices))
    if reverse:
        zipped = reversed(zipped)
    # zipped.sort(key=lambda t: sum([v for v in t[0]]))
    seen_pairs = set()
    for dst, indexes in zipped:
        if tuple(sorted(list(indexes))) in seen_pairs:
            continue
        seen_pairs.add(tuple(sorted(list(indexes))))
        print(dst)
        print(indexes)
        Calls, Galls = [], []
        for index in indexes:
            A = graphs_adj[index]
            Call, Gall = loukas.attempt_coarsening('variation_neighborhood', A, k, r)
            if len(Call) == 1:
                break
            Calls.append(Call)
            Galls.append(Gall)
        if len(Galls) < n_neighbors:
            continue
        if print_pyramid:
            keep = 2
            for idx, (Call, Gall) in enumerate(list(zip(Calls, Galls))[:keep]):
                coords = None
                if 'MNIST' in dataset_name or 'DLA' in dataset_name:
                    features = graphs_features[indexes[idx]]
                    coords = retrieve_coordinates(features)
                loukas.print_pyramid(Call, Gall, depth=None, init_coordinates=coords)
        else:
            if 'MNIST' in dataset_name or 'DLA' in dataset_name:
                attributes = [graphs_features[index] for index in indexes]
            else:
                attributes = None
            print_graphs(Calls, Galls, attributes)
        plt.show()
        to_stop = input('Tape something... ')
        if to_stop == 'stop':
            break


if __name__ == '__main__':
    seed = random.randint(1, 1000 * 1000)
    print('Use seed %d'%seed)
    np.random.seed(seed + 3165)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='Task to execute. Only %s are currently available.'%str(dataset.available_tasks()))
    parser.add_argument('--cached', default=False, type=bool)
    parser.add_argument('--reverse', default=False, type=bool)
    parser.add_argument('--pyramid', default=False, type=bool)
    args = parser.parse_args()
    if args.task in dataset.available_tasks():
        with tf.device('/cpu'):
            find_nearest_neighbor(args.task, n_neighbors=7,
                                  load_neighbors=args.cached,
                                  reverse=args.reverse,
                                  print_pyramid=args.pyramid)
    elif args.task == 'synthetic':
        num_nodes = 10
        edges = [(0,1),(1,2),(2,3),(3,4),(4,0),(3,5),(5,6),(6,7),(7,8),(8,6)]
        A = np.zeros(shape=(num_nodes, num_nodes))
        for (i,j) in edges:
            A[i,j] = 1
            A[j,i] = 1
        A = tf.constant(A,dtype=tf.int64)
        k, r = 18, 0.99
        Call, Gall = loukas.attempt_coarsening('variation_neighborhood', A, k, r)
        loukas.print_pyramid(Call, Gall, depth=None)
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()