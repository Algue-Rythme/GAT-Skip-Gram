import argparse
import os
import random
import numpy as np
import tensorflow as tf
import dataset
import skip_gram
import gat
import gcn


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

def get_graph_wl_extractor(extractor, max_depth, num_features):
    if extractor == 'gat':
        num_heads = 4
        assert num_features%num_heads == 0
        F = num_features // num_heads
        return gat.StackedGraphAttention(max_depth, num_heads=num_heads, num_features=F)
    if extractor == 'gcn':
        return gcn.StackedGraphConvolution(max_depth, num_features)
    raise ValueError

def train_dense(dataset_name, extractor, max_depth, num_features, k, num_epochs, lbda, train_wl):
    graph_adj, graph_features = dataset.read_dortmund(dataset_name, standardize=True)
    num_graphs = len(graph_adj)
    wl_embedder = get_graph_wl_extractor(extractor, max_depth, num_features)
    wl_embedder.trainable = train_wl
    graph_embedder = skip_gram.GraphEmbedding(num_graphs, num_features)
    wl_embedder_file, graph_embedder_file, csv_file = get_weight_filenames(dataset_name)
    num_batchs = num_graphs
    for epoch in range(num_epochs):
        print('epoch %d/%d'%(epoch+1, num_epochs))
        skip_gram.train_epoch_dense(wl_embedder, graph_embedder, graph_adj, graph_features, max_depth, k, num_batchs, lbda)
        wl_embedder.save_weights(wl_embedder_file)
        graph_embedder.save_weights(graph_embedder_file)
        graph_embedder.dump_to_csv(csv_file)

if __name__ == '__main__':
    seed = random.randint(1, 1000 * 1000)
    print('Seed used: %d', seed)
    np.random.seed(seed + 789)
    tf.random.set_seed(seed + 146)
    available_tasks = ['ENZYMES', 'PROTEINS', 'PROTEINS_full', 'MUTAG', 'PTC_FM', 'NCI1', 'PTC_FR']
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='Task to execute. Only %s are currently available.'%str(available_tasks))
    args = parser.parse_args()
    if args.task in available_tasks:
        train_dense(args.task, extractor='gcn',
                    max_depth=2, num_features=256, k=1,
                    num_epochs=30, lbda=4., train_wl=False)
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
