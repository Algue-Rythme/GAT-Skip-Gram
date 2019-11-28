"""Graph Convolutional Neural Network.
"""

import tensorflow as tf


def normalize_adjacency(A):
    A = A + tf.eye(num_rows=A.shape[0])
    D = tf.math.reduce_sum(A, axis=1)
    D = tf.linalg.diag(tf.math.rsqrt(D))
    A = D @ A @ D
    return A


class GraphConvolution(tf.keras.layers.Layer):

    def __init__(self,
                 F_,
                 auto_normalize=False,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        self.F_ = F_  # Number of output features (F' in the paper)
        self.auto_normalize = auto_normalize
        self.activation = tf.keras.activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernel = None       # Layer kernel
        self.bias = None        # Layer bias

        self.edge_kernel = None

        self.output_dim = self.F_

        super(GraphConvolution, self).__init__(**kwargs)


    def build(self, input_shape):
        assert len(input_shape) in [2, 3]
        F = input_shape[0][-1]

        # Layer kernel
        self.kernel = self.add_weight(shape=(F, self.F_),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      name='kernel')

        # Layer bias
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.F_, ),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        name='bias')

        if len(input_shape) == 3 and input_shape[2] != (0,):
            F_edge = input_shape[2][-1]

            # Layer kernel for edges
            self.edge_kernel = self.add_weight(shape=(F_edge, self.F_),
                                               initializer=self.kernel_initializer,
                                               regularizer=self.kernel_regularizer,
                                               constraint=self.kernel_constraint,
                                               name='edge_kernel')

    def call(self, inputs):
        X = inputs[0]  # Node features (N x F)
        A = inputs[1]  # normalized Adjacency matrix (N x N)
        if self.auto_normalize:
            A = normalize_adjacency(A)
        y = A @ X @ self.kernel
        if self.edge_kernel is not None:
            Z = inputs[2]
            y = y + tf.einsum('ij,ijf,fg->ig', A, Z, self.edge_kernel)
        if self.use_bias:
            y = y + self.bias
        y = self.activation(y)
        return y


class StackedGraphConvolution(tf.keras.models.Model):

    def __init__(self, num_layers, num_features, last_layer_only):
        super(StackedGraphConvolution, self).__init__()
        self.num_layers = num_layers
        self.last_layer_only = last_layer_only
        self.ga_layers = [GraphConvolution(num_features, activation='linear') for _ in range(num_layers)]
        self.activation = tf.keras.layers.Activation('relu')

    def vocab_size(self):
        if self.last_layer_only:
            return 1
        return self.num_layers

    def call(self, inputs):
        x = inputs[0]
        A = normalize_adjacency(inputs[1])
        outputs = []
        for index, layer in enumerate(self.ga_layers):
            x = layer([x, A] + inputs[2:])
            if not self.last_layer_only or index+1 == len(self.ga_layers):
                outputs.append(x)
            x = self.activation(x)
        return outputs
