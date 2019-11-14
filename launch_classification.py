import argparse
import os
import random
import numpy as np
import tensorflow as tf
import dataset
import kron


def train_test_split(*lsts, **kwargs):
    len_x = len(lsts[0])
    assert all([len_x == len(x) for x in lsts])
    indices = tf.range(start=0, limit=len_x, dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    shuffled_lsts = [[x[index] for index in shuffled_indices] for x in lsts]
    train_lim = int(kwargs['split_ratio'] * len_x)
    train = [x[train_lim:] for x in shuffled_lsts]
    test = [x[:train_lim] for x in shuffled_lsts]
    x_train, y_train = train[:-1], train[-1]
    x_test, y_test = test[:-1], test[-1]
    nester = lambda xs: list(map(lambda t: tuple(map(lambda x: tf.constant(x, dtype=tf.float32), t)), zip(*xs)))
    x_train, x_test = nester(x_train), nester(x_test)
    return (x_train, y_train), (x_test, y_test)

def train_single_epoch(x_train, y_train, model, optimizer):
    num_graphs = len(y_train)
    progbar = tf.keras.utils.Progbar(num_graphs)
    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    for batch in range(num_graphs):
        with tf.GradientTape() as tape:
            logits = model(x_train[batch])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_train[batch], logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        metric.update_state(y_train[batch], tf.nn.softmax(logits))
        progbar.update(batch+1, [('loss', float(loss.numpy().mean())), ('acc', metric.result().numpy())])

def evaluate(x_test, y_test, model):
    num_graphs = len(y_test)
    progbar = tf.keras.utils.Progbar(num_graphs)
    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    for batch in range(num_graphs):
        logits = model(x_test[batch])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_test[batch], logits)
        metric.update_state(y_test[batch], tf.nn.softmax(logits))
        progbar.update(batch+1, [('loss', float(loss.numpy().mean())), ('acc', metric.result().numpy())])

def shuffle_dataset(x_train, y_train):
    indices = list(range(len(y_train)))
    random.shuffle(indices)
    return [x_train[index] for index in indices], [y_train[index] for index in indices]

def train_classification(dataset_name, num_epochs):
    with tf.device('/gpu:0'):
        graph_adj, graph_features, _ = dataset.read_dortmund(dataset_name,
                                                             with_edge_features=False,
                                                             standardize=True)
        labels, num_labels = dataset.read_graph_labels(dataset_name)
        (x_train, y_train), (x_test, y_test) = train_test_split(graph_features, graph_adj, labels, split_ratio=0.2)
        del graph_adj, graph_features, _, labels
        model = kron.ConvolutionalCoarsenerNetwork(output_dim=num_labels, num_stages=1, num_features=256, activation='relu')
        optimizer = tf.keras.optimizers.Adam()
        for epoch in range(num_epochs):
            print('Epoch %d/%d'%(epoch+1, num_epochs))
            train_single_epoch(x_train, y_train, model, optimizer)
            evaluate(x_test, y_test, model)
            model.save_weights(os.path.join(dataset_name+'_weights', 'coarsener.h5'))
            shuffle_dataset(x_train, y_train)
            print('')


if __name__ == '__main__':
    seed = random.randint(1, 1000 * 1000)
    print('Seed used: %d'%seed)
    np.random.seed(seed + 789)
    tf.random.set_seed(seed + 146)
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='Task to execute. Only %s are currently available.'%str(dataset.available_tasks()))
    args = parser.parse_args()
    if args.task in dataset.available_tasks():
        train_classification(args.task, num_epochs=30)
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
