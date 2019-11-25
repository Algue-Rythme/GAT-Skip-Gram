import argparse
import os
import random
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import dataset


def learn_embeddings(embeddings, labels, ratio, algo):
    test_size = int(labels.shape[0] * ratio)
    x_train, x_test, y_train, y_test = train_test_split(embeddings, labels, test_size=test_size, shuffle=True)
    if algo[:3] == 'knn':
        model = KNeighborsClassifier(n_neighbors=int(algo[3:]))
    elif algo == 'adaboost':
        model = AdaBoostClassifier(n_estimators=100)
    elif algo[:4] == 'svm-':
        model = svm.SVC(gamma='scale', kernel=algo[4:])
    elif algo == 'perceptron':
        y_train = tf.keras.utils.to_categorical(y_train)
        y_train = y_train[:,y_train.any(0)]
        num_classes = y_train.shape[-1]
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='linear'),
            tf.keras.layers.Activation('softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        model.fit(x_train, y_train, epochs=10, verbose=0, batch_size=64)
        y_test = tf.keras.utils.to_categorical(y_test)
        y_test = y_test[:,y_test.any(0)]
        _, acc = model.evaluate(x_test, y_test, verbose=0)
        return acc
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    cur_acc = accuracy_score(y_test, y_pred)
    return cur_acc

def evaluate_embeddings(dataset_name, num_tests):
    graph_embedder_filename = os.path.join(dataset_name+'_weights', 'graph_embeddings.csv')
    labels_filename = os.path.join(dataset_name, '%s_graph_labels.txt'%dataset_name)
    with open(graph_embedder_filename, 'r') as f:
        embeddings_data = np.loadtxt(f, delimiter='\t').astype(np.float32)
    with open(labels_filename, 'r') as f:
        labels_data = np.loadtxt(f, ndmin=1)
    accs = []
    for algo in ['perceptron', 'svm-sigmoid', 'svm-poly', 'svm-rbf', 'knn3', 'knn7', 'adaboost']:
        print('Algo %s'%algo)
        cur_num_tests = num_tests
        if algo == 'adaboost':
            cur_num_tests = max(2, (num_tests // 20))
        if algo == 'perceptron':
            cur_num_tests = max(2, (num_tests // 10))
        progbar = tf.keras.utils.Progbar(cur_num_tests)
        for test in range(cur_num_tests):
            acc = learn_embeddings(embeddings_data, labels_data, ratio=0.2, algo=algo)
            if algo == 'svm-rbf':
                accs.append(acc)
            progbar.update(test+1, [('acc', acc*100.)])
    acc_avg = tf.math.reduce_mean(accs)
    acc_std = tf.math.reduce_std(accs)
    return acc_avg, acc_std

if __name__ == '__main__':
    seed = random.randint(1, 1000 * 1000)
    print('Use seed %d'%seed)
    np.random.seed(seed + 3165)
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='Task to execute. Only %s are currently available.'%str(dataset.available_tasks()))
    args = parser.parse_args()
    if args.task in dataset.available_tasks():
        acc_avg_g, acc_std_g = evaluate_embeddings(args.task, num_tests=100)
        print('Accuracy: %.2f+-%.2f%%'%(acc_avg_g*100., acc_std_g*100.))
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
