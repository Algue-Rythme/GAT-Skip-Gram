import argparse
import os
import random
import numpy as np
import tensorflow as tf
import dataset
import skip_gram
import gat
import gcn
import kron
import loukas
import svm
import utils


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

def get_graph_wl_extractor(extractor, max_depth, num_features, last_layer_only):
    if extractor == 'gat':
        num_heads = 4
        assert num_features%num_heads == 0
        F = num_features // num_heads
        model = gat.StackedGraphAttention(max_depth, num_heads=num_heads, num_features=F, last_layer_only=last_layer_only)
        return model
    if extractor == 'gcn' or extractor == 'random_matrix':
        model = gcn.StackedGraphConvolution(max_depth, num_features=num_features, last_layer_only=last_layer_only)
        if extractor == 'random_matrix':
            model.trainable = False
        return model
    raise ValueError

def get_graph_embedder_extractor(embedder_extractor, num_graphs, num_features):
    if embedder_extractor == 'raw_embedding':
        return skip_gram.GraphEmbedding(num_graphs, num_features)
    elif embedder_extractor == 'kron':
        return kron.ConvolutionalKronCoarsener(output_dim=num_features, num_stages=2,
                                               num_features=num_features, activation='relu')
    elif embedder_extractor == 'loukas':
        return loukas.ConvolutionalLoukasCoarsener(output_dim=num_features, num_stages=2,
                                                   num_features=num_features,
                                                   coarsening_method='variation_neighborhood',
                                                   pooling_method='mean', block_layer='gcn')
    raise ValueError

def train_embeddings(dataset_name, wl_extractor, embedder_extractor,
                     max_depth, num_features, k, num_epochs, lbda, last_layer_only):
    graph_adj, graph_features, edge_features = dataset.read_dortmund(dataset_name,
                                                                     with_edge_features=False,
                                                                     standardize=True)
    num_graphs = len(graph_adj)
    wl_embedder = get_graph_wl_extractor(wl_extractor, max_depth, num_features, last_layer_only)
    graph_embedder = get_graph_embedder_extractor(embedder_extractor, num_graphs, num_features)
    wl_embedder_file, graph_embedder_file, csv_file = get_weight_filenames(dataset_name)
    num_batchs = num_graphs
    for epoch in range(num_epochs):
        print('epoch %d/%d'%(epoch+1, num_epochs))
        lr = 0.002 * np.math.pow(1.1, - 50.*(epoch / num_epochs))
        skip_gram.train_epoch(
            wl_embedder, graph_embedder,
            graph_features, graph_adj, edge_features,
            k, num_batchs, lbda, lr)
        wl_embedder.save_weights(wl_embedder_file)
        graph_embedder.save_weights(graph_embedder_file)
        graph_embedder.dump_to_csv(csv_file, (graph_features, graph_adj))
        acc_avg, acc_std = svm.evaluate_embeddings(dataset_name, num_tests=10)
        print('Accuracy: %.2f+-%.2f%%'%(acc_avg*100., acc_std*100.))
        print('')


if __name__ == '__main__':
    seed = random.randint(1, 1000 * 1000)
    print('Seed used: %d', seed)
    np.random.seed(seed + 789)
    tf.random.set_seed(seed + 146)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='Task to execute. Only %s are currently available.'%str(dataset.available_tasks()))
    parser.add_argument('--wl_extractor', default='gcn', help='Wesfeiler Lehman extractor. \'gcn\', \'gat\' or \'random_matrix\'')
    parser.add_argument('--embedder_extractor', default='raw_embedding', help='Extractor of graph embeddings. \'raw_embedding\', \'kron\' or \'loukas\'')
    parser.add_argument('--max_depth', type=int, default=4, help='Depth of extractor.')
    parser.add_argument('--num_features', type=int, default=1024, help='Size of feature space')
    parser.add_argument('--k', type=int, default=1, help='Ratio between positive and negative samples')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lbda', type=float, default=1., help='Weight for positive samples')
    parser.add_argument('--last_layer_only', type=bool, default=False, help='Use only vocabulary of biggest radius.')
    args = parser.parse_args()
    print(utils.str_from_args(args))
    if args.task in dataset.available_tasks():
        train_embeddings(args.task, args.wl_extractor, args.embedder_extractor,
                         args.max_depth, args.num_features, args.k,
                         args.num_epochs, args.lbda, args.last_layer_only)
        acc, std = svm.evaluate_embeddings(args.task, num_tests=100)
        utils.record_args(args.task, args, acc, std)
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
