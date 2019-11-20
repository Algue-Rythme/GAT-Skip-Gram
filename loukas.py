import numpy as np
import tensorflow as tf
from pygsp import graphs
import loukas_coarsening.coarsening_utils as loukas
import gcn


class ConvolutionalLoukasCoarsener(tf.keras.models.Model):

    def __init__(self, output_dim, num_stages, num_features, coarsening_method, pooling_method):
        super(ConvolutionalLoukasCoarsener, self).__init__()
        self.k = 6
        self.r = 0.99
        self.coarsening_method = coarsening_method
        self.pooling_method = pooling_method
        self.output_dim = output_dim
        self.fc_in = tf.keras.layers.Dense(num_features, activation='relu')
        self.blocks = [gcn.GraphConvolution(num_features, auto_normalize=True, activation='relu')
                       for _ in range(num_stages)]
        self.fc_out = tf.keras.layers.Dense(output_dim, activation='linear')

    def pooling(self, coarsening_matrix, X):
        assert self.pooling_method == 'mean'
        coarsening_matrix = coarsening_matrix.power(2)
        coarsening_matrix = tf.constant(coarsening_matrix.todense(), dtype=tf.float32)
        return coarsening_matrix @ X

    def call_block(self, coarsening_matrix, G_reduced, X):
        A_reduced = G_reduced.W.todense().astype(dtype=np.float32)
        for block in self.blocks:
            X = block((X, A_reduced))
        if coarsening_matrix is not None:
            X = self.pooling(coarsening_matrix, X)
        return X, A_reduced

    def call(self, inputs):
        X, A = inputs
        G = graphs.Graph(A.numpy())
        _, _, Call, Gall = loukas.coarsen(G, K=self.k, r=self.r, method=self.coarsening_method)
        Call.append(None)
        X = self.fc_in(X)
        assert len(Call) == len(Gall) and Gall
        for coarsening_matrix, G_reduced in zip(Call, Gall):
            X, A = self.call_block(coarsening_matrix, G_reduced, X)
        X = tf.math.reduce_sum(X, axis=-2, keepdims=True)
        X = self.fc_out(X)
        return tf.squeeze(X)
