import argparse
import os
import random
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def learn_embeddings(graph_embedder_file, labels_file, ratio):
    with open(graph_embedder_file, 'r') as f:
        embeddings = np.loadtxt(f, delimiter='\t').astype(np.float32)
    with open(labels_file, 'r') as f:
        labels = np.loadtxt(f, ndmin=1)
    test_size = int(labels.shape[0] * ratio)
    x_train, x_test, y_train, y_test = train_test_split(embeddings, labels, test_size=test_size, shuffle=True)
    clf = svm.SVC(gamma='scale')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print('Accuracy: ', acc)


if __name__ == '__main__':
    random.seed(599)
    np.random.seed(3165)
    available_tasks = ['ENZYMES', 'PROTEINS', 'PROTEINS_full']
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='Task to execute. Only %s are currently available.'%str(available_tasks))
    args = parser.parse_args()
    if args.task in available_tasks:
        graph_embedder_filename = os.path.join(args.task+'_weights', 'graph_embeddings.csv')
        labels_filename = os.path.join(args.task, '%s_graph_labels.txt'%args.task)
        learn_embeddings(graph_embedder_filename, labels_filename, ratio=0.2)
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
