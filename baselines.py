import argparse
import collections
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
import utils


def cosine_similarity_histogram(embeddings):
    epsilon = 1e-4
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + epsilon)
    scores = embeddings @ np.transpose(embeddings)
    if not (-1.1 <= np.min(scores) <= np.max(scores) <= 1.1):
        print(np.min(scores), np.max(scores))
    scores = scores[np.triu_indices(embeddings.shape[0])]
    probs, buckets = np.histogram(scores, bins=20, range=(-1., 1.))
    return probs / scores.shape[0], buckets

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
        model.fit(x_train, y_train, epochs=20, verbose=0, batch_size=64)
        y_test = tf.keras.utils.to_categorical(y_test)
        y_test = y_test[:,y_test.any(0)]
        _, acc = model.evaluate(x_test, y_test, verbose=0)
        return acc
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    cur_acc = accuracy_score(y_test, y_pred)
    return cur_acc

def reduce_num_tests(algo, num_tests, num_data):
    if algo.startswith('svm-') or algo.startswith('knn'):
        cur_num_tests = min(10, num_tests / max(1, num_data // 1000))
    if algo == 'adaboost':
        cur_num_tests = max(2, (num_tests // 20))
    if algo == 'perceptron':
        cur_num_tests = max(3, (num_tests // 5))
    return cur_num_tests

def evaluate_embeddings(dataset_name, num_tests, final=False, low_memory=False, graph2vec=False):
    embeddings_data, labels_data = utils.get_data(dataset_name, graph2vec)
    accs = collections.defaultdict(list)
    algos = ['svm-rbf']
    if final:
        algos = ['svm-sigmoid', 'svm-poly', 'svm-rbf', 'knn3', 'knn7', 'adaboost']
        if not low_memory:
            algos.append('perceptron')
    # num_data = int(labels_data.shape[0])
    for algo in algos:
        print('Algo %s'%algo)
        # cur_num_tests = reduce_num_tests(algo, num_tests, num_data)
        cur_num_tests = 10
        progbar = tf.keras.utils.Progbar(cur_num_tests)
        for test in range(cur_num_tests):
            acc = learn_embeddings(embeddings_data, labels_data, ratio=0.2, algo=algo)
            accs[algo].append(acc)
            progbar.update(test+1, [('acc', acc*100.)])
    probs, bins = cosine_similarity_histogram(embeddings_data)
    for prob, bucket_a, bucket_b in zip(probs.tolist(), bins.tolist(), bins.tolist()[1:]):
        print('[%.2f,%.2f]=%.2f'%(bucket_a, bucket_b, prob), end=' ')
    print('')
    acc_avg, acc_std = dict(), dict()
    for key in accs:
        acc_avg[key] = tf.math.reduce_mean(accs[key])
        acc_std[key] = tf.math.reduce_std(accs[key])
    return acc_avg, acc_std

if __name__ == '__main__':
    seed = random.randint(1, 1000 * 1000)
    print('Use seed %d'%seed)
    np.random.seed(seed + 3165)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='Task to execute. Only %s are currently available.'%str(dataset.available_tasks()))
    parser.add_argument('--origin', help='Whether we should use the CSV of Graph2Vec or Vanilla', default='vanilla')
    args = parser.parse_args()
    if args.task in dataset.available_tasks():
        graph2vec_b = args.origin == 'graph2vec'
        acc_avg_g, acc_std_g = evaluate_embeddings(args.task, num_tests=100, graph2vec=graph2vec_b)
        print('Accuracy: %.2f+-%.2f%%'%(acc_avg_g*100., acc_std_g*100.))
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
