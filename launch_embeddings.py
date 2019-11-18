import argparse
import os
import random
import numpy as np
import tensorflow as tf
import dataset
import skip_gram
import gat
import gcn
import svm


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
        return gat.StackedGraphAttention(max_depth, num_heads=num_heads, num_features=F, last_layer_only=last_layer_only)
    if extractor == 'gcn':
        return gcn.StackedGraphConvolution(max_depth, num_features=num_features, last_layer_only=last_layer_only)
    raise ValueError

def train_embeddings(dataset_name, extractor, max_depth, num_features, k, num_epochs, lbda, train_wl, last_layer_only):
    graph_adj, graph_features, edge_features = dataset.read_dortmund(dataset_name,
                                                                     with_edge_features=False,
                                                                     standardize=True)
    num_graphs = len(graph_adj)
    wl_embedder = get_graph_wl_extractor(extractor, max_depth, num_features, last_layer_only)
    wl_embedder.trainable = train_wl
    graph_embedder = skip_gram.GraphEmbedding(num_graphs, num_features)
    wl_embedder_file, graph_embedder_file, csv_file = get_weight_filenames(dataset_name)
    num_batchs = num_graphs
    for epoch in range(num_epochs):
        print('epoch %d/%d'%(epoch+1, num_epochs))
        lr = np.math.pow(1.1, -0.5 * epoch) * 0.02
        skip_gram.train_epoch(
            wl_embedder, graph_embedder,
            graph_adj, graph_features, edge_features,
            k, num_batchs, lbda, lr)
        wl_embedder.save_weights(wl_embedder_file)
        graph_embedder.save_weights(graph_embedder_file)
        graph_embedder.dump_to_csv(csv_file)
        acc_avg, acc_std = svm.evaluate_embeddings(dataset_name, num_tests=10)
        print('Accuracy: %.2f+-%.2f%%'%(acc_avg*100., acc_std*100.))


if __name__ == '__main__':
    seed = random.randint(1, 1000 * 1000)
    print('Seed used: %d', seed)
    np.random.seed(seed + 789)
    tf.random.set_seed(seed + 146)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='Task to execute. Only %s are currently available.'%str(dataset.available_tasks()))
    parser.add_argument('--extractor',default='gcn', help='Wesfeiler Lehman extractor. \'gcn\' or \'gat\'')
    parser.add_argument('--max_depth', type=int, default=3, help='Depth of extractor.')
    parser.add_argument('--num_features', type=int, default=1024, help='Size of feature space')
    parser.add_argument('--k', type=int, default=1, help='Ratio between positive and negative samples')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lbda', type=float, default=1., help='Weight for positive samples')
    args = parser.parse_args()
    if args.task in dataset.available_tasks():
        train_embeddings(args.task, args.extractor,
                         args.max_depth, args.num_features, args.k,
                         args.num_epochs, args.lbda, train_wl=True, last_layer_only=True)
        acc, std = svm.evaluate_embeddings(args.task, num_tests=100)
        experiments_filename = os.path.join(args.task+'_weights', 'experiments.txt')
        args_dict = vars(args)
        args_formatted = ' '.join([key+'='+str(args_dict[key]) for key in sorted(args_dict.keys())])
        acc_formatted = (' acc=%.2f' % (acc*100)) + (' std=%.2f' % (std*100))
        with open(experiments_filename, 'a') as experiments_file:
            experiments_file.write(args_formatted + acc_formatted + '\n')
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
