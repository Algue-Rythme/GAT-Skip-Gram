import argparse
import logging
import math
import os
import random
import traceback
import numpy as np
import tensorflow as tf
import baselines
import dataset
import skip_gram
import gat
import gcn
import kron
import loukas
import truncated_krylov
import utils


def get_graph_wl_extractor(extractor, max_depth, num_features, last_layer_only):
    if extractor == 'gat':
        num_heads = 1
        assert num_features%num_heads == 0
        F = num_features // num_heads
        model = gat.StackedGraphAttention(max_depth,
                                          num_heads=num_heads,
                                          num_features=F,
                                          last_layer_only=last_layer_only)
        return model
    if extractor.startswith('gcn') or extractor == 'random_matrix':
        rooted_subtree = 'rooted' in extractor
        model = gcn.StackedGraphConvolution(max_depth,
                                            num_features=num_features,
                                            last_layer_only=last_layer_only,
                                            rooted_subtree=rooted_subtree)
        if extractor == 'random_matrix':
            model.trainable = False
        return model
    if extractor == 'krylov':
        num_hops = 4
        model = truncated_krylov.TruncatedKrylov(max_depth,
                                                 num_features=num_features,
                                                 num_hops=num_hops,
                                                 last_layer_only=last_layer_only)
        return model
    raise ValueError

def get_graph_embedder_extractor(embedder_extractor, num_graphs, num_features):
    if embedder_extractor == 'raw_embedding':
        return skip_gram.GraphEmbedding(num_graphs, num_features)
    elif embedder_extractor == 'kron':
        return kron.ConvolutionalKronCoarsener(output_dim=num_features, num_stages=2,
                                               num_features=num_features, activation='relu')
    elif embedder_extractor.startswith('loukas'):
        block_layer = [name for name in ['gcn', 'gat', 'krylov'] if name in embedder_extractor][0]
        return loukas.ConvolutionalLoukasCoarsener(output_dim=num_features, num_stages=2,
                                                   num_features=num_features,
                                                   coarsening_method='variation_neighborhood',
                                                   pooling_method='mean', block_layer=block_layer)
    raise ValueError

def train_embeddings(dataset_name, graph_inputs, wl_extractor, embedder_extractor,
                     max_depth, num_features, k, num_epochs, lbda, last_layer_only):
    num_graphs = len(graph_inputs[0])
    wl_embedder = get_graph_wl_extractor(wl_extractor, max_depth, num_features,
                                         last_layer_only)
    graph_embedder = get_graph_embedder_extractor(embedder_extractor, num_graphs, num_features)
    wl_embedder_file, graph_embedder_file, csv_file = utils.get_weight_filenames(dataset_name)
    num_batchs = math.ceil(num_graphs // (k+1))
    for epoch in range(num_epochs):
        print('epoch %d/%d'%(epoch+1, num_epochs))
        lr = 0.002 * np.math.pow(1.1, - 50.*(epoch / num_epochs))
        skip_gram.train_epoch(
            wl_embedder, graph_embedder,
            graph_inputs,
            k, num_batchs, lbda, lr)
        if epoch+1 == num_epochs or (epoch+1)%5 == 0:
            wl_embedder.save_weights(wl_embedder_file)
            graph_embedder.save_weights(graph_embedder_file)
            graph_embedder.dump_to_csv(csv_file, graph_inputs)
            acc, std = baselines.evaluate_embeddings(dataset_name, num_tests=10)
            print('Accuracy: %.2f+-%.2f%%'%(acc*100., std*100.))
            print('')


if __name__ == '__main__':
    seed = random.randint(1, 1000 * 1000)
    print('Seed used: %d'%seed)
    np.random.seed(seed + 789)
    tf.random.set_seed(seed + 146)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='Task to execute. Only %s are currently available.'%str(dataset.available_tasks()))
    parser.add_argument('--wl_extractor', default='gcn', help='Wesfeiler Lehman extractor. \'gcn\', \'gat\', \'random_matrix\' or \'krylov\'')
    parser.add_argument('--embedder_extractor', default='raw_embedding', help='Extractor of graph embeddings. \'raw_embedding\', \'kron\' or \'loukas\'')
    parser.add_argument('--max_depth', type=int, default=1, help='Depth of extractor.')
    parser.add_argument('--num_features', type=int, default=1024, help='Size of feature space')
    parser.add_argument('--k', type=int, default=1, help='Ratio between positive and negative samples')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lbda', type=float, default=1., help='Weight for positive samples')
    parser.add_argument('--last_layer_only', type=bool, default=False, help='Use only vocabulary of biggest radius.')
    parser.add_argument('--num_tests', type=int, default=10, help='Number of repetitions')
    parser.add_argument('--device', default='0', help='Index of the target GPU')
    args = parser.parse_args()
    departure_time = utils.get_now()
    print(departure_time)
    if args.task in dataset.available_tasks():
        with tf.device('/gpu:'+args.device):
            graph_inputs = dataset.read_dortmund(args.task,
                                                 with_edge_features=False,
                                                 standardize=True)
            accs = []
            num_tests = args.num_tests
            for test in range(num_tests):
                print('Test %d'%(test+1))
                print(utils.str_from_args(args))
                restart = True
                while restart:
                    try:
                        train_embeddings(args.task, graph_inputs, args.wl_extractor, args.embedder_extractor,
                                         args.max_depth, args.num_features, args.k,
                                         args.num_epochs, args.lbda, args.last_layer_only)
                        cur_acc, _ = baselines.evaluate_embeddings(args.task, num_tests=60, final=True, low_memory=True)
                        restart = False
                    except Exception as e:
                        print(e.__doc__)
                        print(e)
                        logging.error(traceback.format_exc())
                        restart = True
                accs.append(cur_acc)
                print('')
            acc_avg = tf.math.reduce_mean(accs)
            acc_std = tf.math.reduce_std(accs)
            print(utils.str_from_args(args))
            print('Final accuracy: %.2f+-%.2f%%'%(acc_avg*100., acc_std*100.))
            utils.record_args('embeddings', departure_time, args.task, args, acc_avg, acc_std)
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
