import tensorflow as tf
import loukas_coarsening.coarsening_utils as loukas


class ConvolutionalLoukasCoarsener(tf.keras.models.Model):

    def __init__(self, output_dim, num_stages, num_features, method):
        super(ConvolutionalLoukasCoarsener, self).__init__()
        self.k = 10
        self.r = 0.99
        self.method = method
        self.output_dim = output_dim
        self.fc_in = tf.keras.layers.Dense(num_features, activation='relu')

    def call(self, inputs):
        X, A = inputs
        C, Gc, Call, Gall = loukas.coarsen(G, K=self.k, r=self.r, method=self.method)
        return None
