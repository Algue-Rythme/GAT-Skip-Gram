import argparse
import logging
import os
import random
import traceback
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
    elif coarsener.startswith('loukas'):
        if 'neighbors' in coarsener:
            variation_method = 'variation_neighborhood'
        elif 'edge' in coarsener:
            variation_method = 'variation_edges'
        else:
            raise ValueError
        block_layer = [name for name in ['gcn', 'gat', 'krylov'] if name in coarsener][0]
        model = loukas.ConvolutionalLoukasCoarsener(output_dim=output_dim, num_stages=num_stages,
                                                    num_features=num_features,
                                                    coarsening_method=variation_method,
                                                    pooling_method='sum', block_layer=block_layer)
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

def train_classification(dataset_name, graph_inputs, coarsener, num_epochs, batch_size, num_stages, num_features):
    try:
        os.mkdir(dataset_name+'_weights')
    except FileExistsError:
        pass
    labels, num_labels = dataset.read_graph_labels(dataset_name)
    (x_train, y_train), (x_val, y_val) = utils.train_test_split(*graph_inputs, labels, split_ratio=0.2)
    (x_val, y_val), (x_test, y_test) = utils.train_test_split(*x_val, y_val, split_ratio=0.5)
    x_train, x_val, x_test = utils.nester(x_train), utils.nester(x_val), utils.nester(x_test)
    del graph_inputs, labels
    model = get_graph_coarsener(coarsener, output_dim=num_labels, num_stages=num_stages, num_features=num_features)
    best_weights = model.get_weights()
    best_epoch, best_acc = 0, 0.
    for epoch in range(num_epochs):
        print('Epoch %d/%d'%(epoch+1, num_epochs))
        lr = 0.0002 * np.math.pow(1.1, - 50.*(epoch / num_epochs))
        optimizer = tf.keras.optimizers.Adam(lr)
        train_single_epoch(x_train, y_train, model, optimizer, batch_size)
        print('Validation set: ')
        acc_val, _ = evaluate(x_val, y_val, model)
        if acc_val >= best_acc:
            best_epoch, best_acc = epoch+1, acc_val
            best_weights = model.get_weights()
            model.save_weights(os.path.join(dataset_name+'_weights', 'coarsener.h5'))
        utils.shuffle_dataset(x_train, y_train)
        print('')
    print('Best model on validation set was found at epoch %d with accuracy %f'%(best_epoch, best_acc))
    model.set_weights(best_weights)
    print('Test set: ')
    acc_test, _ = evaluate(x_test, y_test, model)
    return acc_test


if __name__ == '__main__':
    seed = random.randint(1, 1000 * 1000)
    print('Seed used: %d'%seed)
    np.random.seed(seed + 789)
    tf.random.set_seed(seed + 146)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='Task to execute. Only %s are currently available.'%str(dataset.available_tasks()))
    parser.add_argument('--coarsener', default='loukas-neighbors', help='Graph coarsener. \'kron\' or \'loukas\'')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of graphs in each batch')
    parser.add_argument('--num_stages', type=int, default=2, help='Number of GCN layers in a single coarsening')
    parser.add_argument('--num_features', type=int, default=256, help='Size of feature space')
    parser.add_argument('--device', default='0', help='Index of the target GPU')
    args = parser.parse_args()
    print(utils.str_from_args(args))
    departure_time = utils.get_now()
    print(departure_time)
    if args.task in dataset.available_tasks():
        with tf.device('/gpu:'+args.device):
            graphs = dataset.read_dataset(args.task,
                                          with_edge_features=False,
                                          standardize=True)
            restart = True
            while restart:
                try:
                    acc = train_classification(args.task, graphs, args.coarsener, args.num_epochs,
                                               args.batch_size, args.num_stages, args.num_features)
                    utils.record_args('classification', departure_time, args.task, args, acc, 0.)
                    restart = False
                except Exception as e:
                    print(e.__doc__)
                    print(e)
                    logging.error(traceback.format_exc())
                    restart = True
            print('Final accuracy: %.2f%%'%(acc*100.))
            print(utils.str_from_args(args))
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
