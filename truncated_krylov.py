"""Implementation of Block Krylov layer from

'Break the Ceiling: Stronger Multi-scale DeepGraph Convolutional Networks'

"""

import tensorflow as tf
import utils


class KrylovBlock(tf.keras.layers.Layer):

    def __init__(self, num_features, num_hops):
        super(KrylovBlock, self).__init__()
        self.num_features = num_features
        self.num_hops = num_hops
        self.fc = tf.keras.layers.Dense(num_features, activation='tanh')

    def build(self, input_shape):
        pass

    def call(self, inputs):
        X = inputs[0]  # Node features (N x F)
        A = inputs[1]  # normalized Adjacency matrix (N x N)
        L = utils.normalize_adjacency(A, rooted_subtree=False, identity=True)
        L_pow_X = X
        H = [L_pow_X]
        for _ in range(self.num_hops - 1):
            L_pow_X = L @ L_pow_X
            H.append(L_pow_X)
        H = tf.concat(H, axis=1)
        H = self.fc(H)
        return H


class TruncatedKrylov(tf.keras.models.Model):

    def __init__(self, num_layers, num_features, num_hops, last_layer_only):
        super(TruncatedKrylov, self).__init__()
        self.num_layers = num_layers
        self.last_layer_only = last_layer_only
        self.krylov_layers = [KrylovBlock(num_features, num_hops) for _ in range(num_layers)]

    def vocab_size(self):
        if self.last_layer_only:
            return 1
        return self.num_layers

    def call(self, inputs):
        x = inputs[0]
        A = inputs[1]
        outputs = []
        for index, layer in enumerate(self.krylov_layers):
            x = layer([x, A] + inputs[2:])
            if not self.last_layer_only or index+1 == len(self.krylov_layers):
                outputs.append(x)
        return outputs
