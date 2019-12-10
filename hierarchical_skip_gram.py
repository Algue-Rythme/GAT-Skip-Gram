import argparse
import math
import random
import numpy as np
import scipy
import tensorflow as tf
import baselines
import dataset
import gcn
import loukas
import utils


class DifferentiablePooler(tf.keras.layers.Layer):

    def __init__(self, num_features, pooling_method):
        super(DifferentiablePooler, self).__init__()
        self.num_features = num_features
        self.pooling_method = pooling_method
        self.gnn_in = gcn.GraphConvolution(self.num_features, activation='relu', auto_normalize=False)
        self.gnn_out = tf.keras.layers.Dense(self.num_features, activation='relu')

    def call(self, inputs):
        X, A, C = inputs[:3]
        A = utils.normalize_adjacency(A, rooted_subtree=False)
        X = self.gnn_in((X, A))
        X = loukas.pooling(self.pooling_method, C, X)
        X = self.gnn_out(X)
        return X


class HierarchicalLoukas(tf.keras.models.Model):

    def __init__(self, num_features, max_num_stages, coarsening_method, pooling_method):
        super(HierarchicalLoukas, self).__init__()
        self.k = 6
        self.r = 0.99
        self.num_features = num_features
        self.max_num_stages = max_num_stages
        self.coarsening_method = coarsening_method
        self.pooling_method = pooling_method
        self.fc_in = tf.keras.layers.Dense(self.num_features, activation='relu')
        self.pooling_layers = []
        self.wl_layers = []
        for _ in range(self.max_num_stages):
            pooling_layer = DifferentiablePooler(self.num_features, self.pooling_method)
            self.pooling_layers.append(pooling_layer)
            wl_layer = gcn.GraphConvolution(self.num_features, auto_normalize=True)
            self.wl_layers.append(wl_layer)

    def max_depth(self):
        return self.max_num_stages

    def features_dim(self):
        return self.num_features

    def coarsen(self, A):
        pooling_matrices, graphs = loukas.attempt_coarsening(self.coarsening_method, A, self.k, self.r)
        del pooling_matrices[-1]
        last_graph = graphs[-1]
        for _ in range(self.max_depth() - len(graphs) + 1):
            pooling_matrices.append(scipy.sparse.identity(int(last_graph.N)))
            graphs.append(last_graph)
        pooling_matrices.append(None)
        return pooling_matrices, graphs

    def call(self, inputs):
        X, A = inputs[:2]
        pooling_matrices, graphs = self.coarsen(A)
        X = self.fc_in(X)
        gen_input = zip(self.pooling_layers, self.wl_layers, pooling_matrices, graphs)
        vocab, context, indicator = [], [], []
        for pooling_layer, wl_layer, C, graph in gen_input:
            A = graph.W.todense().astype(dtype=np.float32)
            wl_X = wl_layer((X, A))
            X = pooling_layer((X, A, C))
            vocab.append(wl_X)
            context.append(X)
            indicator.append(C.todense().astype(dtype=np.float32))
        return vocab, context, indicator


def get_current_labels(graph_lengths, depth, k, indicator):
    context_size = int(indicator[depth][k].shape[0])
    before = tf.zeros(shape=(context_size, sum(graph_lengths[   :k])), dtype=tf.float32)
    after =  tf.zeros(shape=(context_size, sum(graph_lengths[k+1: ])), dtype=tf.float32)
    onehot = tf.dtypes.cast(tf.dtypes.cast(indicator[depth][k], dtype=tf.bool), dtype=tf.float32)
    labels = tf.concat([before, onehot, after], axis=1)
    return labels


def forward_batch(model, graph_inputs, batch_size):
    graph_indexes = random.sample(list(range(len(graph_inputs[0]))), batch_size+1)
    max_depth = model.max_depth()
    vocab = [[] for _ in range(max_depth)]
    context = [[] for _ in range(max_depth)]
    indicator = [[] for _ in range(max_depth)]
    for graph_index in graph_indexes:
        vocab_g, context_g, indicator_g = model(utils.dispatch(graph_inputs, graph_index))
        for depth in range(max_depth):
            vocab[depth].append(vocab_g[depth])
            context[depth].append(context_g[depth])
            indicator[depth].append(indicator_g[depth])
    graph_lengths = [int(graph_inputs[0][index].shape[0]) for index in graph_indexes]  # TODO: indexed by depth
    return graph_lengths, vocab, context, indicator


def update_metric(metric, labels, similarity):
    labels = tf.reshape(labels, [-1])
    similarity = tf.nn.sigmoid(tf.reshape(similarity, [-1]))
    metric.update_state(labels, similarity)


def process_batch(model, graph_inputs, batch_size, lbda, metrics):
    graph_lengths, vocab, context, indicator = forward_batch(model, graph_inputs, batch_size)
    max_depth = model.max_depth()
    losses = [[] for _ in range(max_depth)]
    for depth in range(max_depth):
        cur_vocab = tf.concat(vocab[depth], axis=0)
        for k in range(batch_size):
            labels = get_current_labels(graph_lengths, depth, k, indicator)
            similarity = tf.einsum('if,jf->ij', context[depth][k], cur_vocab)
            loss = tf.nn.weighted_cross_entropy_with_logits(labels, similarity, lbda*float(batch_size))
            losses[depth].append(tf.math.reduce_mean(loss))
            update_metric(metrics[depth], labels, similarity)
        losses[depth] = tf.math.reduce_mean(losses[depth])
    return losses


def train_epoch(model, graph_inputs,
                batch_size, num_batchs, lbda, lr):
    optimizer = tf.keras.optimizers.Adam(lr)
    progbar = tf.keras.utils.Progbar(num_batchs)
    metrics = [tf.keras.metrics.BinaryAccuracy() for _ in range(model.max_depth())]
    for step in range(num_batchs):
        with tf.GradientTape() as tape:
            losses = process_batch(model, graph_inputs, batch_size, lbda, metrics)
            total_loss = tf.math.reduce_mean(losses)
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        progbar.update(step+1, [
            ('loss_%d'%(i+1), float(loss.numpy().mean()))
            for i, loss in enumerate(losses)] + [
                ('acc_%d'%(i+1), float(metric.result().numpy()))
                for i, metric in enumerate(metrics)])


def train_embeddings(dataset_name, 
                     max_depth, num_features, batch_size,
                     num_epochs, lbda, verbose):
    graph_inputs = dataset.read_dortmund(dataset_name,
                                         with_edge_features=False,
                                         standardize=True)
    num_graphs = len(graph_inputs[0])
    model = HierarchicalLoukas(num_features=num_features,
                               max_num_stages=max_depth,
                               coarsening_method='variation_neighborhood',
                               pooling_method='mean')
    _, graph_embedder_file, csv_file = utils.get_weight_filenames(dataset_name)
    num_batchs = math.ceil(num_graphs // (batch_size+1))
    for epoch in range(num_epochs):
        print('epoch %d/%d'%(epoch+1, num_epochs))
        lr = 0.002 * np.math.pow(1.1, - 50.*(epoch / num_epochs))
        train_epoch(model, graph_inputs,
                    batch_size, num_batchs, lbda, lr)
        if epoch+1 == num_epochs or (epoch+1)%5 == 0 or verbose == 1:
            model.save_weights(graph_embedder_file)
            model.dump_to_csv(csv_file, graph_inputs)
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
    parser.add_argument('--max_depth', type=int, default=1, help='Depth of extractor.')
    parser.add_argument('--num_features', type=int, default=128, help='Size of feature space')
    parser.add_argument('--batch_size', type=int, default=8, help='Size of batchs')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lbda', type=float, default=1., help='Weight for positive samples')
    parser.add_argument('--num_tests', type=int, default=10, help='Number of repetitions')
    parser.add_argument('--device', default='0', help='Index of the target GPU')
    parser.add_argument('--verbose', type=int, default=0, help='0 or 1.')
    args = parser.parse_args()
    departure_time = utils.get_now()
    print(departure_time)
    if args.task in dataset.available_tasks():
        with tf.device('/gpu:'+args.device):
            accs = []
            num_tests = args.num_tests
            for test in range(num_tests):
                print('Test %d'%(test+1))
                print(utils.str_from_args(args))
                train_embeddings(args.task, args.max_depth, args.num_features,
                                 args.batch_size,args.num_epochs, args.lbda, args.verbose)
                cur_acc, _ = baselines.evaluate_embeddings(args.task, num_tests=60, final=True, low_memory=True)
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