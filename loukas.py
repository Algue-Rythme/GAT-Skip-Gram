import warnings
import numpy as np
import scipy
import tensorflow as tf
from pygsp import graphs
import loukas_coarsening.coarsening_utils as loukas
import gat
import gcn


class ConvolutionalLoukasCoarsener(tf.keras.models.Model):

    def __init__(self, output_dim, num_stages, num_features, coarsening_method, pooling_method, block_layer):
        super(ConvolutionalLoukasCoarsener, self).__init__()
        self.k = 6
        self.r = 0.99
        self.coarsening_method = coarsening_method
        self.pooling_method = pooling_method
        self.output_dim = output_dim
        self.fc_in = tf.keras.layers.Dense(num_features, activation='relu')
        self.blocks = []
        for _ in range(num_stages):
            if block_layer == 'gcn':
                block = gcn.GraphConvolution(num_features, auto_normalize=True, activation='relu')
            elif block_layer == 'gat':
                block = gat.GraphAttention(num_features, attn_heads=2, attn_heads_reduction='average')
            else:
                raise ValueError
            self.blocks.append(block)
        self.fc_middle = tf.keras.layers.Dense(num_features, activation='relu')
        self.fc_out = tf.keras.layers.Dense(output_dim, activation='linear')

    def get_embeddings_from_indices(self, inputs, indices):
        return [self([input[index] for input in inputs])[0] for index in indices]

    def get_weights_from_indices(self, _):
        return self.trainable_variables

    def dump_to_csv(self, csv_file, inputs):
        with open(csv_file, 'w') as f:
            for graph_input in zip(*inputs):
                embed, _ = self(graph_input)
                f.write('\t'.join(map(str, embed.numpy().tolist()))+'\n')

    def pooling(self, coarsening_matrix, X):
        coarsening_matrix = coarsening_matrix.power(2)
        coarsening_matrix = tf.constant(coarsening_matrix.todense(), dtype=tf.float32)
        if self.pooling_method == 'mean':
            return coarsening_matrix @ X
        elif self.pooling_method == 'sum':
            X = coarsening_matrix @ X
            mask = tf.not_equal(coarsening_matrix, tf.constant(0., dtype=tf.float32))
            indicator = tf.dtypes.cast(mask, tf.float32)
            X = X * tf.reduce_sum(indicator, axis=-1, keepdims=True)
            return X
        elif self.pooling_method == 'max':
            X = tf.einsum('nm,mf->nmf', coarsening_matrix, X)
            X = tf.math.reduce_max(X, axis=-2)
            return X
        else:
            raise ValueError

    def call_block(self, coarsening_matrix, G_reduced, X):
        A_reduced = G_reduced.W.todense().astype(dtype=np.float32)
        for block in self.blocks:
            X = block((X, A_reduced))
        if coarsening_matrix is not None:
            X = self.pooling(coarsening_matrix, X)
        return X, A_reduced

    def attempt_coarsening(self, A):
        G = graphs.Graph(A.numpy())
        attempt = 1
        while attempt > 0:
            try:
                _, _, Call, Gall = loukas.coarsen(G, K=self.k, r=self.r, method=self.coarsening_method)
                attempt = 0
            except scipy.sparse.linalg.ArpackError as e:
                warnings.warn('attempt %d: '%attempt+str(e))
                attempt += 1
        Call.append(None)
        return Call, Gall

    def call(self, inputs):
        X, A = inputs
        Call, Gall = self.attempt_coarsening(A)
        X = self.fc_in(X)
        assert len(Call) == len(Gall) and Gall
        for coarsening_matrix, G_reduced in zip(Call, Gall):
            X, A = self.call_block(coarsening_matrix, G_reduced, X)
        Xs = tf.math.reduce_sum(X, axis=-2, keepdims=True)
        Xm = tf.math.reduce_max(X, axis=-2, keepdims=True)
        X = self.fc_middle(tf.concat([Xs, Xm], axis=-1))
        X = self.fc_out(X)
        return tf.squeeze(X), {'pyramid_depth':len(Gall), 'last_level_width':Gall[-1].N}
