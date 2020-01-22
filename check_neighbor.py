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

def print_graphs(Calls, Galls):
    size = 3
    alpha=0.55
    edge_width=0.8
    node_size=20
    fig = plt.figure(figsize=(len(Galls)*size*3, size))
    for i, (Call, Gall) in enumerate(zip(Calls, Galls)):
        G = Gall[0]
        C = Call[0]
        C = C.toarray()
        # print(G, C)
        G.set_coordinates(kind='spring')
        # pygsp.plotting.plot_graph(G)
        ax = fig.add_subplot(1, len(Galls), i+1)
        ax.axis('off')
        [x,y] = G.coords.T
        edges = np.array(G.get_edge_list()[0:2])
        for eIdx in range(0, edges.shape[1]):
            ax.plot(x[edges[:,eIdx]], y[edges[:,eIdx]], color='k', alpha=alpha, lineWidth=edge_width)
        for i in range(int(C.shape[0])):
            subgraph = np.arange(G.N)[C[i,:] > 0]
            ax.scatter(x[subgraph], y[subgraph], c='b', s=node_size, alpha=alpha)


def find_nearest_neighbor(dataset_name, n_neighbors, print_pyramid=True):
    X, Y = utils.get_data(dataset_name)
    print('Compute Nearest Neighbors')
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
    distances, indices = nn.kneighbors(X)
    all_graphs = dataset.read_dataset(args.task,
                                      with_edge_features=False,
                                      standardize=True)
    ADJ_IDX = 1
    graphs_adj = all_graphs[ADJ_IDX]
    k = 18
    r = 0.99
    zipped = list(zip(distances, indices))
    # zipped.sort(key=lambda t: sum([v for v in t[0]]))
    seen_pairs = set()
    for dst, indexes in zipped:
        # if dst.sum() < 1.:
        #     continue
        # A1 = graphs_adj[indexes[0]]
        # A2 = graphs_adj[indexes[1]]
        # if A1.shape == A2.shape and np.abs((A1 - A2).numpy()).sum() < 1.:
        #     continue
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
            # if len(Gall) <= 3:
            #    continue
            Calls.append(Call)
            Galls.append(Gall)
            # print(all_graphs[0][index])
            # print(all_graphs[1][index])
        if len(Galls) < n_neighbors:
            continue
        if print_pyramid:
            for Call, Gall in zip(Calls, Galls):
                loukas.print_pyramid(Call, Gall, depth=None)
        else:
            print_graphs(Calls, Galls)
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
    args = parser.parse_args()
    if args.task in dataset.available_tasks():
        find_nearest_neighbor(args.task, n_neighbors=7, print_pyramid=False)
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()