import tensorflow as tf
import pygsp
import matplotlib as plt
import gcn


def get_schur_blocks(matrix, major, minor):
    AB = tf.gather(matrix, indices=major, axis=-2)
    CD = tf.gather(matrix, indices=minor, axis=-2)
    A = tf.gather(AB, indices=major, axis=-1)
    B = tf.gather(AB, indices=minor, axis=-1)
    C = tf.gather(CD, indices=major, axis=-1)
    D = tf.gather(CD, indices=minor, axis=-1)
    return A, B, C, D

def get_schur_complement(matrix, major, minor):
    A, B, C, D = get_schur_blocks(matrix, major, minor)
    M_D = A - B @ tf.linalg.inv(D) @ C
    return M_D

def build_laplacian(adj):
    degrees = tf.linalg.diag(tf.math.reduce_sum(adj, axis=0))
    laplacian = degrees - adj
    return laplacian

def largest_eigen_vector_method(laplacian):
    _, eigenvectors = tf.linalg.eigh(tf.dtypes.cast(laplacian, dtype=tf.float64))
    largest_eigenvector = eigenvectors[:,-1]
    polarity = tf.math.sign(largest_eigenvector).numpy().tolist()
    major = [index for index, sign in enumerate(polarity) if sign > 0]
    minor = [index for index, sign in enumerate(polarity) if sign < 0]
    if not major:
        return minor[:len(minor)//2], minor[len(minor)//2:]
    if not minor:
        return major[:len(major)//2], major[len(major)//2:]
    return major, minor

def kron_reduction(laplacian, major, minor):
    l_reduced = get_schur_complement(laplacian, major, minor)
    l_reduced = (l_reduced + tf.transpose(l_reduced)) / 2.  # symmetrize
    adj = tf.eye(num_rows=l_reduced.shape[0]) - tf.math.sign(l_reduced)
    return adj

class KronCoarsening(tf.keras.layers.Layer):

    def __init__(self, vertex_selection='largest_eigenvector'):
        super(KronCoarsening, self).__init__()
        assert vertex_selection in ['largest_eigenvector']
        self.reduction_method = largest_eigen_vector_method

    def build(self, _):
        pass

    def call(self, inputs):
        X, A = inputs
        L = build_laplacian(A)
        major, minor = self.reduction_method(L)
        X_reduced = tf.gather(X, indices=major, axis=-2)
        A_reduced = kron_reduction(L, major, minor)
        return X_reduced, A_reduced

def plot_graph(A_pyramid):
    for adj in A_pyramid:
        graph = pygsp.graphs.Graph(adj)
        graph.set_coordinates()
        pygsp.plotting.plot_graph(graph)
    plt.pyplot.show()

class ConvolutionalCoarsenerNetwork(tf.keras.models.Model):

    def __init__(self, output_dim, num_stages, num_features, activation):
        super(ConvolutionalCoarsenerNetwork, self).__init__()
        self.output_dim = output_dim
        self.fc_in = tf.keras.layers.Dense(num_features, activation='relu')
        self.blocks = [gcn.GraphConvolution(num_features, auto_normalize=True, activation=activation)
                       for _ in range(num_stages)]
        self.coarsener = KronCoarsening()
        self.fc_out = tf.keras.layers.Dense(output_dim, activation='linear')

    def call(self, inputs):
        try:
            X, A = inputs
            A_pyramid = [A]
            X = self.fc_in(X)
            while X.shape[0] != 1:
                for block in self.blocks:
                    X = block((X, A))
                X, A = self.coarsener((X, A))
                A_pyramid.append(A)
            X = self.fc_out(X)
            return tf.squeeze(X)
        except tf.errors.InvalidArgumentError as e:
            print('\n', e)
            plot_graph(A_pyramid)
            raise e
