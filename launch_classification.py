import argparse
import os
import random
import numpy as np
import tensorflow as tf
import dataset
import loukas
import kron
import utils


def get_graph_coarsener(coarsener, output_dim, num_stages, num_features):
    if coarsener == 'kron':
        model = kron.ConvolutionalKronCoarsener(output_dim=output_dim, num_stages=num_stages,
                                                num_features=num_features, activation='relu')
        return model
    elif coarsener == 'loukas':
        model = loukas.ConvolutionalLoukasCoarsener(output_dim=output_dim, num_stages=num_stages,
                                                    num_features=num_features,
                                                    coarsening_method='variation_neighborhood',
                                                    pooling_method='sum', block_layer='gcn')
        return model
    raise ValueError

def train_single_epoch(x_train, y_train, model, optimizer, batch_size):
    num_graphs = len(y_train)
    progbar = tf.keras.utils.Progbar(num_graphs)
    for batch in range(0, num_graphs, batch_size):
        acc_loss = None
        with tf.GradientTape() as tape:
            max_batch = min(batch+batch_size, num_graphs)
            for sample in range(batch, max_batch):
                logits, infos = model(x_train[sample])
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_train[batch], logits)
                acc_loss = loss if sample == batch else acc_loss + loss
                accuracy = (y_train[batch] == float(tf.math.argmax(tf.nn.softmax(logits)).numpy()))
                progbar.update(sample+1, [('loss', float(loss.numpy().mean())), ('acc', accuracy)] + list(infos.items()))
            acc_loss = acc_loss / tf.constant(max_batch - batch, dtype=tf.float32)
        gradients = tape.gradient(acc_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def evaluate(x_test, y_test, model):
    num_graphs = len(y_test)
    progbar = tf.keras.utils.Progbar(num_graphs)
    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    accs = []
    for batch in range(num_graphs):
        logits, infos = model(x_test[batch])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_test[batch], logits)
        metric.update_state(y_test[batch], tf.nn.softmax(logits))
        acc_avg = metric.result().numpy()
        accs.append(acc_avg)
        progbar.update(batch+1, [('loss', float(loss.numpy().mean())), ('acc', acc_avg)] + list(infos.items()))
    return tf.math.reduce_mean(accs), tf.math.reduce_std(accs)

def train_classification(dataset_name, coarsener, num_epochs, batch_size, num_stages, num_features):
    try:
        os.mkdir(dataset_name+'_weights')
    except FileExistsError:
        pass
    with tf.device('/gpu:0'):
        graph_adj, graph_features, _ = dataset.read_dortmund(dataset_name,
                                                             with_edge_features=False,
                                                             standardize=True)
        labels, num_labels = dataset.read_graph_labels(dataset_name)
        (x_train, y_train), (x_test, y_test) = utils.train_test_split(graph_features, graph_adj, labels, split_ratio=0.2)
        del graph_adj, graph_features, _, labels
        model = get_graph_coarsener(coarsener, output_dim=num_labels, num_stages=num_stages, num_features=num_features)
        acc_avg, acc_std = 0., 0.
        for epoch in range(num_epochs):
            print('Epoch %d/%d'%(epoch+1, num_epochs))
            lr = 0.002 * np.math.pow(1.1, - 50.*(epoch / num_epochs))
            optimizer = tf.keras.optimizers.Adam(lr)
            train_single_epoch(x_train, y_train, model, optimizer, batch_size)
            acc_avg, acc_std = evaluate(x_test, y_test, model)
            model.save_weights(os.path.join(dataset_name+'_weights', 'coarsener.h5'))
            utils.shuffle_dataset(x_train, y_train)
            print('')
        return acc_avg, acc_std


if __name__ == '__main__':
    seed = random.randint(1, 1000 * 1000)
    print('Seed used: %d'%seed)
    np.random.seed(seed + 789)
    tf.random.set_seed(seed + 146)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='Task to execute. Only %s are currently available.'%str(dataset.available_tasks()))
    parser.add_argument('--coarsener', default='loukas', help='Graph coarsener. \'kron\' or \'loukas\'')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of graphs in each batch')
    parser.add_argument('--num_stages', type=int, default=2, help='Number of GCN layers in a single coarsening')
    parser.add_argument('--num_features', type=int, default=256, help='Size of feature space')
    args = parser.parse_args()
    print(utils.str_from_args(args))
    if args.task in dataset.available_tasks():
        acc, std = train_classification(args.task, args.coarsener, args.num_epochs,
                                        args.batch_size, args.num_stages, args.num_features)
        utils.record_args(args.task, args, acc, std)
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
