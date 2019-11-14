import argparse
import os
import random
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import dataset


def learn_embeddings(embeddings, labels, ratio):
    test_size = int(labels.shape[0] * ratio)
    x_train, x_test, y_train, y_test = train_test_split(embeddings, labels, test_size=test_size, shuffle=True)
    clf = svm.SVC(gamma='scale')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    cur_acc = accuracy_score(y_test, y_pred)
    return cur_acc

def evaluate_embeddings(dataset_name, num_tests):
    graph_embedder_filename = os.path.join(dataset_name+'_weights', 'graph_embeddings.csv')
    labels_filename = os.path.join(dataset_name, '%s_graph_labels.txt'%dataset_name)
    with open(graph_embedder_filename, 'r') as f:
        embeddings_data = np.loadtxt(f, delimiter='\t').astype(np.float32)
    with open(labels_filename, 'r') as f:
        labels_data, _ = np.loadtxt(f, ndmin=1)
    accs = []
    progbar = tf.keras.utils.Progbar(num_tests)
    for test in range(num_tests):
        acc = learn_embeddings(embeddings_data, labels_data, ratio=0.2)
        accs.append(acc)
        progbar.update(test+1, [('acc', acc*100.)])
    acc_avg = tf.math.reduce_mean(accs)
    acc_std = tf.math.reduce_std(accs)
    print('Accuracy: %.2f+-%.2f%%'%(acc_avg*100., acc_std*100.))


if __name__ == '__main__':
    seed = random.randint(1, 1000 * 1000)
    print('Use seed %d'%seed)
    np.random.seed(seed + 3165)
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='Task to execute. Only %s are currently available.'%str(dataset.available_tasks()))
    args = parser.parse_args()
    if args.task in dataset.available_tasks():
        evaluate_embeddings(args.task, num_tests=100)
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
