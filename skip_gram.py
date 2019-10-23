import itertools
import random
import tensorflow as tf


class GraphEmbedding(tf.keras.layers.Layer):

    def __init__(self, num_graphs, num_features):
        super(GraphEmbedding, self).__init__()
        self.num_graphs = num_graphs
        self.num_features = num_features
        self.embed = None

    def build(self):
        self.embed = self.add_weight(name='graph_embed',
                                     shape=[self.num_graphs, self.num_features])

    def call(self, indices):
        return tf.gather(self.embed, indices)


def get_labels(pair_per_graph, k):
    labels = []
    for i in range(k+1):
        a = [0.]*(i*pair_per_graph)
        b = [1.]*(pair_per_graph)
        c = [0.]*((k-i)*pair_per_graph)
        labels.append(a+b+c)
    labels = tf.constant(labels, dtype=tf.float32)
    return labels


def get_batch_embeddings(wl_embedder, graph_embedder,
                         graph_adj, graph_f,
                         pair_per_graph, k):
    graph_indexes = random.sample(list(range(len(graph_adj))), k+1)
    graph_embeds = graph_embedder(graph_indexes)
    nodes_tensor = []
    for index in graph_indexes:
        node_embeds = wl_embedder(graph_adj[index], graph_f[index])
        num_nodes = int(tf.shape(graph_adj[index])[0])
        max_depth = len(node_embeds)
        vocab = itertools.product(range(num_nodes),range(max_depth))
        vocab = random.sample(list(vocab), pair_per_graph)
        node_embeds = [node_embeds[depth][node,:] for depth, node in vocab]
        nodes_tensor += node_embeds
    nodes_tensor = tf.stack(nodes_tensor)
    return nodes_tensor, graph_embeds


def train_epoch(wl_embedder, graph_embedder,
                graph_adj, graph_f,
                pair_per_graph, k,
                num_batchs):
    labels = get_labels(pair_per_graph, k)
    optimizer = tf.keras.optimizers.Adam()
    progbar = tf.keras.utils.Progbar(num_batchs)
    for step in range(num_batchs):
        with tf.GradientTape() as tape:
            nodes_tensor, graph_embeds = get_batch_embeddings(
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
        progbar.update(step+1, [('loss', float(loss.numpy()))])
