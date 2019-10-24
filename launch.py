import argparse
import random
import numpy as np
import tensorflow as tf
import dataset
import skip_gram
import gat


def train_dense(dataset_name, max_depth, k, num_batchs):
    graph_adj, graph_features = dataset.read_dortmund(dataset_name, standardize=True)
    num_graphs = len(graph_adj)
    num_heads, num_features = 4, 256
    wl_embedder = gat.StackedGraphAttention(max_depth, num_heads, num_features)
    embedding_size = num_heads * num_features
    graph_embedder = skip_gram.GraphEmbedding(num_graphs, embedding_size)
    skip_gram.train_epoch_dense(wl_embedder, graph_embedder, graph_adj, graph_features, max_depth, k, num_batchs)


if __name__ == '__main__':
    random.seed(515)
    np.random.seed(789)
    tf.random.set_seed(146)
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='Task to execute. ENZYMES is the only task available.')
    args = parser.parse_args()
    if args.task == 'ENZYMES':
        train_dense('ENZYMES', max_depth=2, k=1, num_batchs=500)
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
