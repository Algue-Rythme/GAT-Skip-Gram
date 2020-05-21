import argparse
import collections
import os
import random
import multiprocessing
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
import tensorflow as tf
import dataset
import utils
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def cosine_similarity_histogram(embeddings):
    epsilon = 1e-4
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + epsilon)
    scores = embeddings @ np.transpose(embeddings)
    if not (-1.1 <= np.min(scores) <= np.max(scores) <= 1.1):
        print(np.min(scores), np.max(scores))
    scores = scores[np.triu_indices(embeddings.shape[0])]
    probs, buckets = np.histogram(scores, bins=20, range=(-1., 1.))
    return probs / scores.shape[0], buckets

def kfold_svm(x_non_test, x_test, y_non_test, y_test, grid):
    kfold = StratifiedKFold(n_splits=5)
    if grid:
        parameters = [{'kernel': ['rbf'], 'gamma':['scale', 0.1, 1, 10, 100, 1000], 'C': [0.1, 1, 10, 100, 1000]}]
    else:
        parameters = [{'kernel': ['rbf'], 'gamma':['scale'], 'C': [1.]}]
    n_jobs = max(multiprocessing.cpu_count()-2, 1)
    grid_search = GridSearchCV(svm.SVC(), parameters, scoring='accuracy',
                               cv=kfold.split(x_non_test, y_non_test), refit=True,
                               verbose=1, n_jobs=n_jobs)
    grid_search.fit(x_non_test, y_non_test)
    train_acc = float(grid_search.best_score_)

    best_model = grid_search.best_estimator_
    y_pred_test = best_model.predict(x_test)
    test_acc = accuracy_score(y_test, y_pred_test)

    return train_acc*100., test_acc*100., dict(grid_search.best_params_)

def load_two_files(dataset_name, graph2vec):
    if graph2vec:  # WARNING SPLIT TRAIN TEST
        embeddings_data, labels = utils.get_data(dataset_name, graph2vec=True)
        y_train, y_test = train_test_split(labels, train_size=0.8, shuffle=True)
    else:
        embeddings_data, labels = utils.get_data(dataset_name, graph2vec=False)
        train_indexes, test_indexes = utils.get_train_test_indexes(dataset_name)
        x_train = embeddings_data[train_indexes]
        x_test = embeddings_data[test_indexes]
        y_train = labels[train_indexes]
        y_test = labels[test_indexes]
        embeddings_data = np.concatenate([x_train, x_test])
    return embeddings_data, y_train, y_test

def evaluate_embeddings(dataset_name, graph2vec=False, normalize='', histo=False, grid=False):
    embeddings_data, y_train, y_test = load_two_files(dataset_name, graph2vec)
    if 'std' in normalize:
        embeddings_data = embeddings_data - np.mean(embeddings_data, axis=0, keepdims=True)
        epsilon = 1e-3
        embeddings_data = embeddings_data / (epsilon + np.std(embeddings_data, axis=0, keepdims=True))
    if 'norm' in normalize:
        embeddings_data = embeddings_data / np.linalg.norm(embeddings_data, axis=1, keepdims=True)
    x_train = embeddings_data[:int(y_train.shape[0]),:]
    x_test = embeddings_data[int(y_train.shape[0]):,:]
    train_accs, test_accs, params = kfold_svm(x_train, x_test, y_train, y_test, grid)
    print('train_accs=%.2f%%'%train_accs)
    print('test_accs=%.2f%%'%test_accs)
    print('params=%s'%str(params))
    if histo:
        probs, bins = cosine_similarity_histogram(embeddings_data)
        for prob, bucket_a, bucket_b in zip(probs.tolist(), bins.tolist(), bins.tolist()[1:]):
            print('[%.2f,%.2f]=%.2f'%(bucket_a, bucket_b, prob), end=' ')
        print('')
    return train_accs, test_accs

if __name__ == '__main__':
    infinity = 1000 * 1000
    first_non_null_natural_integer = 1
    seed = random.randint(first_non_null_natural_integer, infinity)
    optimal_seed_shift_obtained_by_extensive_grid_search = 3165
    np.random.seed(seed + optimal_seed_shift_obtained_by_extensive_grid_search)
    print('Use seed %d'%seed)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='Task to execute. Only %s are currently available.'%str(dataset.available_tasks()))
    parser.add_argument('--origin', help='Whether we should use the CSV of Graph2Vec or Vanilla', default='vanilla')
    parser.add_argument('--normalize', help='Normalize ?', default='std')
    args = parser.parse_args()
    if args.task in dataset.available_tasks():
        graph2vec_b = args.origin == 'graph2vec'
        trains, tests = evaluate_embeddings(args.task, graph2vec=graph2vec_b, normalize=args.normalize)
        print('train_acc=%.2f%% test_acc=%.2f%%'%(trains*100., tests*100.))
    else:
        print('Unknown task %s'%args.task)
        parser.print_help()
