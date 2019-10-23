import argparse
import tensorflow as tf
import dataset
import skip_gram
import gat


def train(dataset_name, pair_per_graph, k, num_batchs):
    graph_adj, graph_features = dataset.read_dortmund(dataset_name, standardize=False)
    num_graphs = len(graph_adj)
    num_heads = 4
    num_features = 256
    wl_embedder = gat.StackedGraphAttention(3, num_heads, num_features)
    embedding_size = num_heads * num_features
    graph_embedder = skip_gram.GraphEmbedding(num_graphs, embedding_size)
    skip_gram.train_epoch(wl_embedder, graph_embedder, graph_adj, graph_features, pair_per_graph, k, num_batchs)


if __name__ == '__main__':
    tf.random.set_seed(146)
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='Task to execute. ENZYMES is the only task available.')
    args = parser.parse_args()
    if args.task == 'ENZYMES':
        train('ENZYMES', pair_per_graph=10, k=5, num_batchs=100)
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
