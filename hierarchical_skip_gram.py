import argparse
import collections
import logging
import math
import random
import traceback
#import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.model_selection import train_test_split, ParameterGrid
import tensorflow as tf
import baselines
import dataset
import gat
import gcn
import truncated_krylov
import loukas
import utils
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


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

class MnistConv(tf.keras.layers.Layer):

    def __init__(self, num_features):
        super(MnistConv, self).__init__()
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))
        self.dropouter = tf.keras.layers.Dropout(0.25)
        self.flattener = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(num_features, activation='relu')

    def call(self, x):
        num_nodes = int(x.shape[0])
        x = tf.concat([x, tf.zeros(shape=(num_nodes, 4), dtype=tf.float32)], axis=1)
        x = tf.reshape(x, shape=(num_nodes, 28, 28, 1))
        #img = x.numpy()[13,:,:,0]
        #print(img.shape)
        #print(img)
        #plt.imshow(img)
        #plt.show()
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.pool_1(x)
        x = self.dropouter(x)
        x = self.flattener(x)
        x = self.dense_1(x)
        return x

class HierarchicalLoukas(tf.keras.models.Model):

    def __init__(self, num_features, max_num_stages, coarsening_method, pooling_method, gnn_type, collapse, mnist_conv):
        super(HierarchicalLoukas, self).__init__()
        self.k = 6
        self.r = 0.99
        self.num_features = num_features
        self.max_num_stages = max_num_stages-1 if collapse else max_num_stages
        self.coarsening_method = coarsening_method
        self.pooling_method = pooling_method
        self.gnn_type = gnn_type
        self.collapse = collapse
        if mnist_conv:
            self.fc_in = MnistConv(self.num_features)
        else:
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

    def dump_to_csv(self, csv_file, graph_inputs, include_max=False):
        print('Dumping to CSV...')
        progbar = tf.keras.utils.Progbar(len(graph_inputs[0]))
        with open(csv_file, 'w') as file:
            for step, graph_input in enumerate(zip(*graph_inputs)):
                _, context, _ = self(graph_input)
                if self.collapse:
                    embed = tf.squeeze(context[-1])
                else:
                    mean_features = [tf.math.reduce_mean(t, axis=0) for t in context]
                    if include_max:
                        max_features = [tf.math.reduce_max(t, axis=0) for t in context]
                    else:
                        max_features = []
                    embed = tf.concat(max_features + mean_features, axis=0)
                file.write('\t'.join(map(str, embed.numpy().tolist()))+'\n')
                progbar.update(step+1)

    def coarsen(self, A):
        pooling_matrices, graphs = loukas.attempt_coarsening(self.coarsening_method, A, self.k, self.r)
        assert pooling_matrices[-1] is None
        del pooling_matrices[-1]
        last_graph = graphs[-1]
        for _ in range(self.max_num_stages - (len(graphs)-1)):
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

def forward_batch(model, graph_inputs, graph_indexes):
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

def process_batch(model, graph_inputs, training_indexes, loss_fn, batch_size, metrics):
    graph_indexes = random.sample(training_indexes, batch_size+1)
    graph_lengths, vocab, context, indicator = forward_batch(model, graph_inputs, graph_indexes)
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

def train_epoch(model, graph_inputs, training_indexes, loss_fn,
                batch_size, num_batchs, lr, print_acc):
    optimizer = tf.keras.optimizers.Adam(lr)
    progbar = tf.keras.utils.Progbar(num_batchs)
    metrics = [tf.keras.metrics.BinaryAccuracy() for _ in range(model.max_depth())]
    for step in range(num_batchs):
        with tf.GradientTape() as tape:
            losses = process_batch(model, graph_inputs, training_indexes, loss_fn, batch_size, metrics)
            total_loss = tf.math.reduce_sum(losses)
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss_logs = [('l%d'%(i+1), float(loss.numpy().mean()))
                     for i, loss in enumerate(losses)]
        acc_logs = [('a%d'%(i+1), float(metric.result().numpy()))
                    for i, metric in enumerate(metrics)]
        progbar.update(step+1, loss_logs + (acc_logs if print_acc else []))


def train_embeddings(dataset_name, load_weights_path, graph_inputs,
                     training_indexes, testing_indexes, fully_inductive,
                     loss_type, max_depth, num_features, batch_size,
                     num_epochs, gnn_type, verbose):
    if fully_inductive:
        epoch_indexes = training_indexes  # Less overfitting
    else:
        epoch_indexes = training_indexes + testing_indexes
    local_num_graphs = len(epoch_indexes)
    mnist_conv = dataset_name == 'FRANKENSTEIN'
    if mnist_conv:
        print('Use: MnistConv')
    model = HierarchicalLoukas(num_features=num_features,
                               max_num_stages=max_depth,
                               coarsening_method='variation_neighborhood',
                               pooling_method='sum',
                               gnn_type=gnn_type,
                               collapse=False,
                               mnist_conv=mnist_conv)
    if load_weights_path is not None:
        _ = model([graph_input[0] for graph_input in graph_inputs])
        model.load_weights(load_weights_path)
    loss_fn = get_loss(loss_type)
    _, graph_embedder_file, csv_file = utils.get_weight_filenames(dataset_name)
    num_batchs = math.ceil(local_num_graphs // (batch_size+1))
    for epoch in range(num_epochs):
        print('epoch %d/%d'%(epoch+1, num_epochs))
        lr = 1e-4 * np.math.pow(1.1, - 50.*(epoch / num_epochs))
        if load_weights_path is None:
            train_epoch(model, graph_inputs, epoch_indexes, loss_fn, batch_size, num_batchs, lr,
                        print_acc=(loss_type == 'negative_sampling'))
        if epoch+1 == num_epochs or (epoch+1)%5 == 0 or verbose == 1:
            model.save_weights(graph_embedder_file)
            model.dump_to_csv(csv_file, graph_inputs, include_max=True)
            trains, tests = baselines.evaluate_embeddings(dataset_name, normalize='std', grid=False)
            print('train_acc=%.2f%% test_acc=%.2f%%'%(trains, tests))
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
    parser.add_argument('--num_features', type=int, default=128, help='Size of feature space')
    parser.add_argument('--batch_size', type=int, default=8, help='Size of batchs')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--gnn_type', default='krylov-4', help='Nature of vocabulary extractor')
    parser.add_argument('--num_tests', type=int, default=3, help='Number of repetitions')
    parser.add_argument('--device', default='0', help='Index of the target GPU. Specify \'cpu\' to disable gpu support.')
    parser.add_argument('--verbose', type=int, default=0, help='0 or 1.')
    parser.add_argument('--load_weights_path', default=None, help='File from which to retrieve weights of the model.')
    parser.add_argument('--fully_inductive', action='store_true', help='retrained model on test set')
    parser.add_argument('--no_grid', action='store_true', help='no grid search')
    args = parser.parse_args()
    departure_time = utils.get_now()
    print(departure_time)
    if args.task in dataset.available_tasks():
        device = '/cpu:0' if args.device == 'cpu' else '/gpu:'+args.device
        with tf.device(device):
            all_graphs = dataset.read_dataset(args.task,
                                              with_edge_features=False,
                                              standardize=True)
            if args.no_grid:
                hyper_params = {'depth': [args.max_depth],
                                'gnn_type': [args.gnn_type],
                                'num_features':[args.num_features]}
            else:
                hyper_params = {'depth': [3, 5],
                                'gnn_type': ['krylov-3', 'krylov-5'],
                                'num_features': [128, 256]}
            best_train_accs, best_test_accs, params, best_train_std, best_test_std = [], [], [], [], []
            num_graphs = len(all_graphs[0])
            strates = utils.get_graph_labels(args.task)
            for hyper_param in ParameterGrid(hyper_params):
                gnn_type, depth, num_features = hyper_param['gnn_type'], hyper_param['depth'], hyper_param['num_features']
                train_accs, test_accs = [], []
                num_tests = args.num_tests
                for test in range(num_tests):
                    training_set, testing_set = train_test_split(list(range(num_graphs)), train_size=0.8, shuffle=True, stratify=strates)
                    utils.print_split_indexes_to_csv(args.task, training_set, testing_set)
                    print('Test %d'%(test+1))
                    print(utils.str_from_args(args))
                    print('Hyper params: ', hyper_param)
                    restart = True
                    while restart:
                        try:
                            train_embeddings(args.task, args.load_weights_path, all_graphs,
                                             training_set, testing_set, args.fully_inductive,
                                             args.loss_type, depth, num_features,
                                             args.batch_size, args.num_epochs, gnn_type,
                                             args.verbose)
                            train_acc, test_acc = baselines.evaluate_embeddings(args.task, normalize='std', grid=True)
                            restart = False
                            train_accs.append(train_acc)
                            test_accs.append(test_acc)
                        except Exception as e:
                            print(e.__doc__)
                            try:
                                print(e)
                            except:
                                pass
                            logging.error(traceback.format_exc())
                            restart = False  #True
                    print('')
                print('Hyper params: ', hyper_param)
                best_train_std.append(float(tf.math.reduce_std(train_accs)))
                best_test_std.append(float(tf.math.reduce_std(test_accs)))
                train_accs = tf.math.reduce_mean(train_accs)
                test_accs = tf.math.reduce_mean(test_accs)
                best_train_accs.append(float(train_accs))
                best_test_accs.append(float(test_accs))
                params.append(hyper_param)
                print(utils.str_from_args(args))
                print('final_train_acc=%.2f%% final_test_acc=%.2f%%'%(train_accs, test_accs))
                utils.record_args('embeddings-'+str(hyper_param), departure_time, args.task, args, train_accs, test_accs)
                print('')
            best_acc_index = np.argmax(best_train_accs)
            print('best_test_acc %.2f+-%.2f%%'%(best_test_accs[best_acc_index], best_train_std[best_acc_index]))
            print('with params ', params[best_acc_index])
            print('with train_acc %.2f+-%.2f%%'%(best_train_accs[best_acc_index], best_test_std[best_acc_index]))
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
