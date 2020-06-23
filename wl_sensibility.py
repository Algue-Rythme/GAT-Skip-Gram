import argparse
from collections import Counter
import hashlib
import random
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import multiprocessing
import networkx as nx
import numpy as np
from tqdm import tqdm


def get_degrees(graph):
    return {node:str(degree) for node, degree in graph.degree}

class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """
    def __init__(self, graph, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = get_degrees(graph)
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in self.features.items()]
        self.per_stage = []

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])]+sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.per_stage.append(list(new_features.values()))
        self.extracted_features = self.extracted_features + list(new_features.values())
        return new_features

    def do_recursions(self, per_stage):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()
        if not per_stage:
            return [Counter(self.extracted_features)]
        return [Counter(stage) for stage in self.per_stage]

def wl_procedure(graph, iterations, per_stage):
    machine = WeisfeilerLehmanMachine(graph, iterations)
    labels = machine.do_recursions(per_stage)
    return labels

def get_graph(graph_type, num_nodes):
    if graph_type == 'tree':
        return nx.full_rary_tree(3, num_nodes)
    elif graph_type == 'barbell':
        return nx.barbell_graph(num_nodes // 2, num_nodes // 2)
    elif graph_type == 'turan':
        return nx.turan_graph(num_nodes, 16)
    elif graph_type == 'wheel':
        return nx.wheel_graph(num_nodes)
    elif graph_type == 'cycle':
        return nx.cycle_graph(num_nodes)
    elif graph_type == 'ladder':
        return nx.ladder_graph(num_nodes // 2)
    raise ValueError

def add_random_edge(graph):
    [node_1, node_2] = random.sample(list(graph.nodes()), 2)
    graph.add_edge(node_1, node_2)

def remove_random_edge(graph):
    node_1, node_2 = random.choice(list(graph.edges()))
    graph.remove_edge(node_1, node_2)

def add_random_edges(graph, num_random_edges, cpy=False):
    if cpy:
        graph = nx.Graph(graph)
    nodes_1 = random.choices(list(graph.nodes()),k=num_random_edges)
    nodes_2 = random.choices(list(graph.nodes()),k=num_random_edges)
    graph.add_edges_from([(node_a, node_b) for node_a, node_b in zip(nodes_1, nodes_2)])
    return graph

def remove_random_edges(graph, num_random_edges, cpy=False):
    if cpy:
        graph = nx.Graph(graph)
    nodes_1 = random.choices(list(graph.nodes()),k=num_random_edges)
    nodes_2 = random.choices(list(graph.nodes()),k=num_random_edges)
    graph.remove_edges_from([(node_a, node_b) for node_a, node_b in zip(nodes_1, nodes_2)])
    return graph

def get_random_graph(graph_type, num_nodes, a_edges, r_edges):
    graph = get_graph(graph_type, num_nodes)
    add_random_edges(graph, a_edges)
    remove_random_edges(graph, r_edges)
    return graph

def get_volume(multiset):
    return sum([num for _, num in multiset.items()])

def get_error(multiset_base_lst, multiset_lst):
    errors = []
    for multiset_base, multiset in zip(multiset_base_lst, multiset_lst):
        intersection = multiset & multiset_base
        union = multiset | multiset_base
        volume_intersection = get_volume(intersection)
        volume_union = get_volume(union)
        errors.append(volume_intersection / volume_union * 100.)
    return np.array(errors)

def progressive_damage_remove(graph_base, iterations, damage_max, per_stage):
    multiset_base = wl_procedure(graph_base, iterations, per_stage)
    graph = nx.Graph(graph_base)
    errors = []
    for _ in range(damage_max):
        remove_random_edge(graph)
        multiset = wl_procedure(graph, iterations, per_stage)
        error = get_error(multiset_base, multiset)
        errors.append(error)
    return np.stack(errors)

def one_damage(graph_type, num_nodes, a_edges, r_edges, iterations, damage_max, per_stage):
    graph_base = get_random_graph(graph_type, num_nodes, a_edges, r_edges)
    return progressive_damage_remove(graph_base, iterations, damage_max, per_stage)

def experiment(graph_type, num_nodes, a_edges, r_edges, max_steps, iterations, damage_max, per_stage, plot_all=False):
    args = [graph_type, num_nodes, a_edges, r_edges, iterations, damage_max, per_stage]
    workers = max(multiprocessing.cpu_count()-2, 1)
    errors_lst = Parallel(n_jobs=workers)(delayed(one_damage)(*args) for _ in tqdm(range(max_steps)))
    if plot_all:
        for error in errors_lst:
            print(error)
    error_avg = np.mean(errors_lst, axis=0).transpose()
    print('Average\n', error_avg)
    return error_avg

if __name__ == '__main__':
    seed = random.randint(1, 1000 * 1000)
    print('Seed used: %d'%seed)
    np.random.seed(seed + 139)
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_type', default='tree')
    parser.add_argument('--num_nodes', type=int, default=500)
    parser.add_argument('--a_edges', type=int, default=0)
    parser.add_argument('--r_edges', type=int, default=0)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--damage_max', type=int, default=10)
    parser.add_argument('--per_stage', action='store_true')
    args = parser.parse_args()
    np.set_printoptions(precision=2)
    error_avg = experiment(args.graph_type, args.num_nodes, args.a_edges, args.r_edges,
                           args.max_steps, args.iterations, args.damage_max, args.per_stage)
    for iteration in range(error_avg.shape[0]):
        plt.plot(np.arange(1, error_avg.shape[1]+1), error_avg[iteration],
                 marker='o', label='stage '+str(iteration+1))
    plt.xlabel('Number of edges removed')
    plt.ylabel('Percentage of labels in common')
    plt.title('%s graph'%args.graph_type)
    plt.axis([0, error_avg.shape[1]+1, 0, 100])
    plt.legend()
    plt.show()
