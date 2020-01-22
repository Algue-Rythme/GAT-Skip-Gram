import argparse
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
import dataset
import loukas
import utils


def find_nearest_neighbor(dataset_name):
    X, Y = utils.get_data(dataset_name)
    print('Compute Nearest Neighbors')
    nn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, indices = nn.kneighbors(X)
    all_graphs = dataset.read_dataset(args.task,
                                      with_edge_features=False,
                                      standardize=True)
    ADJ_IDX = 1
    graphs_adj = all_graphs[ADJ_IDX]
    k = 6
    r = 0.99
    zipped = list(zip(distances, indices))
    zipped.sort(key=lambda t: t[0][1])
    seen_pairs = set()
    for dst, indexes in zipped:
        if dst.sum() < 1.:
            continue
        A1 = graphs_adj[indexes[0]]
        A2 = graphs_adj[indexes[1]]
        if A1.shape == A2.shape and np.abs((A1 - A2).numpy()).sum() < 1.:
            continue
        if tuple(indexes) in seen_pairs:
            continue
        seen_pairs.add(tuple(indexes)[::-1])
        print(dst)
        print(indexes)
        for index in indexes:
            A = graphs_adj[index]
            Call, Gall = loukas.attempt_coarsening('variation_neighborhood', A, k, r)
            if len(Gall) <= 3:
                continue
            print(all_graphs[0][index])
            print(all_graphs[1][index])
            loukas.print_pyramid(Call, Gall, depth=None)
        to_stop = input('Tape something... ')
        if to_stop == 'stop':
            break
        plt.show()


if __name__ == '__main__':
    seed = random.randint(1, 1000 * 1000)
    print('Use seed %d'%seed)
    np.random.seed(seed + 3165)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='Task to execute. Only %s are currently available.'%str(dataset.available_tasks()))
    args = parser.parse_args()
    if args.task in dataset.available_tasks():
        find_nearest_neighbor(args.task)
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()