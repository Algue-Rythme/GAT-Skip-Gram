import argparse
import os
import random
import numpy as np
import tensorflow as tf
import dataset
import skip_gram
import gat


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

def train_dense(dataset_name, max_depth, num_heads, num_features, k, num_batchs, num_epochs):
    graph_adj, graph_features = dataset.read_dortmund(dataset_name, standardize=True)
    num_graphs = len(graph_adj)
    wl_embedder = gat.StackedGraphAttention(max_depth, num_heads, num_features)
    embedding_size = num_heads * num_features
    graph_embedder = skip_gram.GraphEmbedding(num_graphs, embedding_size)
    wl_embedder_file, graph_embedder_file, csv_file = get_weight_filenames(dataset_name)
    for epoch in range(num_epochs):
        print('epoch %d/%d'%(epoch+1, num_epochs))
        skip_gram.train_epoch_dense(wl_embedder, graph_embedder, graph_adj, graph_features, max_depth, k, num_batchs)
        wl_embedder.save_weights(wl_embedder_file)
        graph_embedder.save_weights(graph_embedder_file)
        graph_embedder.dump_to_csv(csv_file)

if __name__ == '__main__':
    random.seed(515)
    np.random.seed(789)
    tf.random.set_seed(146)
    available_tasks = ['ENZYMES', 'PROTEINS', 'PROTEINS_full']
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='Task to execute. Only %s are currently available.'%str(available_tasks))
    args = parser.parse_args()
    if args.task in available_tasks:
        train_dense(args.task, max_depth=2, num_heads=4, num_features=32, k=3, num_batchs=100, num_epochs=40)
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
