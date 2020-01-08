import argparse
import logging
import math
import random
import traceback
import numpy as np
import scipy
import tensorflow as tf
import baselines
import dataset
import gat
import gcn
import truncated_krylov
import loukas
import utils


def get_gnn(num_features, gnn_type):
    if gnn_type == 'gcn':
        return gcn.GraphConvolution(num_features, activation='tanh', auto_normalize=True)
    elif gnn_type.startswith('krylov'):
        num_hops = int(gnn_type.split('-')[1])
        return truncated_krylov.KrylovBlock(num_features=num_features, num_hops=num_hops)
    elif gnn_type == 'gat':
        return gat.GraphAttention(num_features, attn_heads=2, attn_heads_reduction='average')
    else:
        raise ValueError


class DifferentiablePooler(tf.keras.layers.Layer):

    def __init__(self, num_features, pooling_method, gnn_type):
        super(DifferentiablePooler, self).__init__()
        self.num_features = num_features
        self.pooling_method = pooling_method
        self.gnn_type = gnn_type
        self.gnn_in = get_gnn(self.num_features, self.gnn_type)
        self.gnn_out = tf.keras.layers.Dense(self.num_features, activation='linear')

    def call(self, inputs):
        X, A, C = inputs[:3]
        X = self.gnn_in((X, A))
        X = loukas.pooling(self.pooling_method, C, X)
        X = self.gnn_out(X)
        return X


class HierarchicalLoukas(tf.keras.models.Model):

    def __init__(self, num_features, max_num_stages, coarsening_method, pooling_method, gnn_type, collapse):
        super(HierarchicalLoukas, self).__init__()
        self.k = 6
        self.r = 0.99
        self.num_features = num_features
        self.max_num_stages = max_num_stages-1 if collapse else max_num_stages
        self.coarsening_method = coarsening_method
        self.pooling_method = pooling_method
        self.gnn_type = gnn_type
        self.collapse = collapse
        self.fc_in = tf.keras.layers.Dense(self.num_features, activation='relu')
        self.pooling_layers = []
        self.wl_layers = []
        for _ in range(self.max_num_stages):
            pooling_layer = DifferentiablePooler(self.num_features, self.pooling_method, self.gnn_type)
            self.pooling_layers.append(pooling_layer)
            wl_layer = get_gnn(self.num_features, self.gnn_type)
            self.wl_layers.append(wl_layer)
        if self.collapse:
            self.fc_middle = tf.keras.layers.Dense(num_features, activation='relu')
            self.fc_out = tf.keras.layers.Dense(num_features, activation='linear')

    def max_depth(self):
        return self.max_num_stages+1 if self.collapse else self.max_num_stages

    def features_dim(self):
        return self.num_features

    def dump_to_csv(self, csv_file, graph_inputs):
        with open(csv_file, 'w') as f:
            for graph_input in zip(*graph_inputs):
                _, context, _ = self(graph_input)
                if self.collapse:
                    embed = tf.squeeze(context[-1])
                else:
                    max_features = [tf.math.reduce_max(t, axis=0) for t in context]
                    mean_features = [tf.math.reduce_mean(t, axis=0) for t in context]
                    embed = tf.concat(max_features + mean_features, axis=0)
                f.write('\t'.join(map(str, embed.numpy().tolist()))+'\n')

    @utils.memoize
    def coarsen(self, A):
        pooling_matrices, graphs = loukas.attempt_coarsening(self.coarsening_method, A, self.k, self.r)
        del pooling_matrices[-1]
        last_graph = graphs[-1]
        for _ in range(self.max_num_stages - len(graphs) + 1):
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
            X = tf.stop_gradient(X)
        if self.collapse:
            vocab.append(X)
            indicator.append(np.zeros(shape=(1, int(X.shape[0])), dtype=np.float32))
            X = self.fc_middle(X)
            Xs = tf.math.reduce_mean(X, axis=-2, keepdims=True)
            Xm = tf.math.reduce_max(X, axis=-2, keepdims=True)
            X = self.fc_out(tf.concat([Xs, Xm], axis=-1))
            context.append(X)
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
    graph_lengths = [0] * max_depth
    for depth in range(max_depth):
        lengths = [int(word.shape[0]) for word in vocab[depth]]
        graph_lengths[depth] = lengths
    return graph_lengths, vocab, context, indicator


def update_metric(metric, labels, similarity):
    labels = tf.reshape(labels, [-1])
    similarity = tf.nn.sigmoid(tf.reshape(similarity, [-1]))
    metric.update_state(labels, similarity)


def negative_sampling(labels, similarity):
    weights = tf.size(labels, out_type=tf.float32) / tf.reduce_sum(labels)
    loss = tf.nn.weighted_cross_entropy_with_logits(labels, similarity, weights)
    return loss


def infoNCE(labels, similarity):
    """InfoNCE objective based on maximization of mutual information.
    """
    similarity = tf.clip_by_value(similarity, -5., 5.)
    similarity = tf.exp(similarity)
    pos_examples = labels * similarity
    pos_examples = tf.math.reduce_sum(pos_examples, axis=-1)

    neg_examples = (1 - labels) * similarity
    neg_examples = tf.math.reduce_sum(neg_examples, axis=-1)

    ratio = pos_examples / neg_examples
    loss = tf.math.log(ratio)
    loss = tf.math.reduce_mean(loss)

    return loss


def get_loss(loss_type):
    if loss_type == 'negative_sampling':
        return negative_sampling
    elif loss_type == 'infoNCE':
        return infoNCE
    raise ValueError


def process_batch(model, graph_inputs, loss_fn, batch_size, metrics):
    graph_lengths, vocab, context, indicator = forward_batch(model, graph_inputs, batch_size)
    max_depth = model.max_depth()
    losses = [[] for _ in range(max_depth)]
    for depth in range(max_depth):
        cur_vocab = tf.concat(vocab[depth], axis=0)
        for k in range(batch_size):
            labels = get_current_labels(graph_lengths[depth], depth, k, indicator)
            similarity = tf.einsum('if,jf->ij', context[depth][k], cur_vocab)
            loss = loss_fn(labels, similarity)
            losses[depth].append(tf.math.reduce_mean(loss))
            update_metric(metrics[depth], labels, similarity)
        losses[depth] = tf.math.reduce_mean(losses[depth])
    return losses


def train_epoch(model, graph_inputs, loss_fn,
                batch_size, num_batchs, lr, print_acc):
    optimizer = tf.keras.optimizers.Adam(lr)
    progbar = tf.keras.utils.Progbar(num_batchs)
    metrics = [tf.keras.metrics.BinaryAccuracy() for _ in range(model.max_depth())]
    for step in range(num_batchs):
        with tf.GradientTape() as tape:
            losses = process_batch(model, graph_inputs, loss_fn, batch_size, metrics)
            total_loss = tf.math.reduce_sum(losses)
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss_logs = [('l%d'%(i+1), float(loss.numpy().mean()))
                     for i, loss in enumerate(losses)]
        acc_logs = [('a%d'%(i+1), float(metric.result().numpy()))
                   for i, metric in enumerate(metrics)]
        progbar.update(step+1, loss_logs + (acc_logs if print_acc else []))


def train_embeddings(dataset_name, graph_inputs, loss_type,
                     max_depth, num_features, batch_size,
                     num_epochs, gnn_type, verbose):
    num_graphs = len(graph_inputs[0])
    model = HierarchicalLoukas(num_features=num_features,
                               max_num_stages=max_depth,
                               coarsening_method='variation_neighborhood',
                               pooling_method='sum',
                               gnn_type=gnn_type,
                               collapse=False)
    loss_fn = get_loss(loss_type)
    _, graph_embedder_file, csv_file = utils.get_weight_filenames(dataset_name)
    num_batchs = math.ceil(num_graphs // (batch_size+1))
    for epoch in range(num_epochs):
        print('epoch %d/%d'%(epoch+1, num_epochs))
        lr = 1e-4 * np.math.pow(1.1, - 50.*(epoch / num_epochs))
        train_epoch(model, graph_inputs, loss_fn, batch_size, num_batchs, lr,
                    print_acc=(loss_type == 'negative_sampling'))
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
    parser.add_argument('--loss_type', default='negative_sampling', help='Loss to minimize. \'negative_sampling\' or \'infoNCE\'')
    parser.add_argument('--max_depth', type=int, default=3, help='Depth of extractor.')
    parser.add_argument('--num_features', type=int, default=256, help='Size of feature space')
    parser.add_argument('--batch_size', type=int, default=32, help='Size of batchs')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gnn_type', default='krylov-4', help='Nature of vocabulary extractor')
    parser.add_argument('--num_tests', type=int, default=25, help='Number of repetitions')
    parser.add_argument('--device', default='0', help='Index of the target GPU')
    parser.add_argument('--verbose', type=int, default=0, help='0 or 1.')
    args = parser.parse_args()
    departure_time = utils.get_now()
    print(departure_time)
    if args.task in dataset.available_tasks():
        with tf.device('/gpu:'+args.device):
            all_graphs = dataset.read_dataset(args.task,
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
                        train_embeddings(args.task, all_graphs, args.loss_type,
                                         args.max_depth, args.num_features,
                                         args.batch_size, args.num_epochs,
                                         args.gnn_type, args.verbose)
                        cur_acc, _ = baselines.evaluate_embeddings(args.task, num_tests=60, final=True, low_memory=True)
                        restart = False
                    except Exception as e:
                        print(e.__doc__)
                        try:
                            print(e)
                        except:
                            pass
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