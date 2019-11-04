import random
import tensorflow as tf


class GraphEmbedding(tf.keras.models.Model):

    def __init__(self, num_graphs, num_features):
        super(GraphEmbedding, self).__init__()
        self.num_graphs = num_graphs
        self.num_features = num_features
        self.embeds = [
            self.add_weight(name=('graph_embed_%d'%i), shape=[self.num_features], initializer='zeros')
            for i in range(self.num_graphs)]

    def dump_to_csv(self, csv_file):
        with open(csv_file, 'w') as f:
            for embed in self.embeds:
                f.write('\t'.join(map(str, embed.numpy().tolist()))+'\n')

    def get_weights_from_indices(self, indices):
        return [self.embeds[index] for index in indices]

    def call(self, indices):
        return tf.stack(self.get_weights_from_indices(indices))

def get_dense_batch(wl_embedder, graph_adj, graph_f, max_depth, k):
    graph_indexes = random.sample(list(range(len(graph_adj))), k+1)
    graph_lengths = [int(graph_adj[index].shape[0])*max_depth for index in graph_indexes]
    nodes_tensor = []
    labels = []
    for i, index in enumerate(graph_indexes):
        node_embeds = wl_embedder([graph_f[index], graph_adj[index]])
        before, now, after = sum(graph_lengths[:i]), graph_lengths[i], sum(graph_lengths[i+1:])
        graph_indicator = before*[0.] + now*[1.] + after*[0.]
        labels.append(graph_indicator)
        nodes_tensor += node_embeds
    nodes_tensor = tf.concat(nodes_tensor, axis=0)
    labels = tf.constant(labels, dtype=tf.float32)
    return nodes_tensor, graph_indexes, labels

def get_updated_metric(metric, labels, similarity, _):
    labels = tf.reshape(labels, [-1])
    similarity = tf.nn.sigmoid(tf.reshape(similarity, [-1]))
    # weights = labels*(k-1) + 1.
    metric.update_state(labels, similarity)
    return metric.result().numpy()

def train_epoch_dense(wl_embedder, graph_embedder,
                      graph_adj, graph_f,
                      max_depth, k, num_batchs):
    optimizer_G = tf.keras.optimizers.SGD(learning_rate=1e-1)
    optimizer_WL = tf.keras.optimizers.Adam()
    progbar = tf.keras.utils.Progbar(num_batchs)
    metric = tf.keras.metrics.BinaryAccuracy()
    for step in range(num_batchs):
        with tf.GradientTape() as tape:
            nodes_tensor, graph_indexes, labels = get_dense_batch(
                wl_embedder, graph_adj, graph_f, max_depth, k)
            graph_embeds = graph_embedder(graph_indexes)
            similarity = tf.einsum('if,jf->ij', graph_embeds, nodes_tensor)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, similarity)
        G_weights = graph_embedder.get_weights_from_indices(graph_indexes)
        WL_weights = wl_embedder.trainable_variables
        dG, dWL = tape.gradient(loss, [G_weights, WL_weights])
        optimizer_G.apply_gradients(zip(dG, G_weights))
        optimizer_WL.apply_gradients(zip(dWL, WL_weights))
        acc = get_updated_metric(metric, labels, similarity, k)
        progbar.update(step+1, [('loss', float(loss.numpy().mean())), ('acc', acc)])
