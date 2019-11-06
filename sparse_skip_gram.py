import itertools
import random
import tensorflow as tf
import dataset
import skip_gram
import gat


def get_sparse_labels(pair_per_graph, k):
    labels = []
    for i in range(k+1):
        a = [0.]*(i*pair_per_graph)
        b = [1.]*(pair_per_graph)
        c = [0.]*((k-i)*pair_per_graph)
        labels.append(a+b+c)
    labels = tf.constant(labels, dtype=tf.float32)
    return labels

def get_sparse_batch(wl_embedder, graph_embedder,
                     graph_adj, graph_f,
                     pair_per_graph, k):
    graph_indexes = random.sample(list(range(len(graph_adj))), k+1)
    graph_embeds = graph_embedder(graph_indexes)
    nodes_tensor = []
    for index in graph_indexes:
        node_embeds = wl_embedder([graph_f[index], graph_adj[index]])
        num_nodes = int(tf.shape(graph_adj[index])[0])
        max_depth = len(node_embeds)
        vocab = itertools.product(range(max_depth),range(num_nodes))
        vocab = random.sample(list(vocab), pair_per_graph)
        node_embeds = [node_embeds[depth][node,:] for depth, node in vocab]
        nodes_tensor += node_embeds
    nodes_tensor = tf.stack(nodes_tensor)
    return nodes_tensor, graph_embeds

def train_epoch_sparse(wl_embedder, graph_embedder,
                       graph_adj, graph_f,
                       pair_per_graph, k,
                       num_batchs):
    labels = get_sparse_labels(pair_per_graph, k)
    optimizer = tf.keras.optimizers.Adam()
    progbar = tf.keras.utils.Progbar(num_batchs)
    for step in range(num_batchs):
        with tf.GradientTape() as tape:
            nodes_tensor, graph_embeds = get_sparse_batch(
                wl_embedder, graph_embedder,
                graph_adj, graph_f,
                pair_per_graph, k)
            similarity = tf.einsum('if,jf->ij', graph_embeds, nodes_tensor)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, similarity)
        G_weights = graph_embedder.trainable_variables
        WL_weights = wl_embedder.trainable_variables
        dG, dWL = tape.gradient(loss, [G_weights, WL_weights])
        optimizer.apply_gradients(zip(dG, G_weights))
        optimizer.apply_gradients(zip(dWL, WL_weights))
        progbar.update(step+1, [('loss', float(loss.numpy().mean()))])

def train_sparse(dataset_name, max_depth, pair_per_graph, k, num_batchs):
    graph_adj, graph_features = dataset.read_dortmund(dataset_name, standardize=False)
    num_graphs = len(graph_adj)
    num_heads = 4
    num_features = 256
    wl_embedder = gat.StackedGraphAttention(max_depth, num_heads, num_features)
    embedding_size = num_heads * num_features
    graph_embedder = skip_gram.GraphEmbedding(num_graphs, embedding_size)
    skip_gram.train_epoch_sparse(wl_embedder, graph_embedder, graph_adj, graph_features, pair_per_graph, k, num_batchs)
