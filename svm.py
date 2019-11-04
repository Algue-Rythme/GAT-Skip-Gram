import argparse
import os
# import random
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def learn_embeddings(embeddings, labels, ratio):
    test_size = int(labels.shape[0] * ratio)
    x_train, x_test, y_train, y_test = train_test_split(embeddings, labels, test_size=test_size, shuffle=True)
    clf = svm.SVC(gamma='scale')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    # y_test = np.zeros(shape=y_test.shape)-1.
    cur_acc = accuracy_score(y_test, y_pred)
    return cur_acc


if __name__ == '__main__':
    # random.seed(599)
    # np.random.seed(3165)
    available_tasks = ['ENZYMES', 'PROTEINS', 'PROTEINS_full', 'MUTAG', 'PTC_FM', 'NCI1', 'PTC_FR']
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='Task to execute. Only %s are currently available.'%str(available_tasks))
    args = parser.parse_args()
    if args.task in available_tasks:
        graph_embedder_filename = os.path.join(args.task+'_weights', 'graph_embeddings.csv')
        labels_filename = os.path.join(args.task, '%s_graph_labels.txt'%args.task)
        with open(graph_embedder_filename, 'r') as f:
            embeddings_data = np.loadtxt(f, delimiter='\t').astype(np.float32)
        with open(labels_filename, 'r') as f:
            labels_data = np.loadtxt(f, ndmin=1)
        num_tests = 100
        accs = []
        for _ in range(num_tests):
            acc = learn_embeddings(embeddings_data, labels_data, ratio=0.2)
            accs.append(acc)
        acc_avg = sum(accs) / len(accs)
        acc_var = sum([(acc-acc_avg)**2 for acc in accs]) / len(accs)
        acc_std = acc_var**0.5
        print('Accuracy: %.2f+-%.2f%%'%(acc_avg*100., acc_std*100.))
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
