"""
Keras models to be used in eager execution
"""

import numpy as np
import tensorflow as tf
tfe = tf.contrib.eager

from utils import extract_q_p, split, safe_concat, join_q_p

# Bijectors: invertible transformations of R^2n. The objects of this type are
# keras models and have an inverse method.
class NICE(tf.keras.Model):
    def __init__(self, shift_model):
        """q,p -> q,p + NN(q).
        NICE bijector for phase space variables."""
        super(NICE, self).__init__()
        self._shift_model = shift_model

    def call(self, x):
        q,p = extract_q_p(x)
        return join_q_p(q, p + self._shift_model(q))

    def inverse(self, x):
        q,p = extract_q_p(x)
        return join_q_p(q, p - self._shift_model(q))

class SymplecticExchange(tf.keras.Model):
    """q,p -> p,-q"""
    def __init__(self):
        super(SymplecticExchange, self).__init__()

    def call(self, x):
        q,p = extract_q_p(x)
        return join_q_p(p,-q)

    def inverse(self, x):
        q,p = extract_q_p(x)
        return join_q_p(-p,q)

class SqueezeAndShift(tf.keras.Model):
    def __init__(self, shift_model):
        """q,p -> q * e^s , e^(-s) * (p + shift_model(q)).
        """
        super(SqueezeAndShift, self).__init__()
        self._shift_model = shift_model
        # scale is a scalar for lattice shift covariance.
        self.scale = tfe.Variable(tf.zeros([1]), name="scale")

    # def build(self, input_size):
    #     # Initialize scale to zeros
    #     self.scale = tfe.Variable(tf.zeros([input_size[1]//2, input_size[2]]),
    #         name="scale")

    def call(self, x):
        q,p = extract_q_p(x)
        return join_q_p(q * tf.exp(self.scale),
                        tf.exp(-self.scale) * (p + self._shift_model(q)))

    def inverse(self, x):
        q,p = extract_q_p(x)
        return join_q_p(q * tf.exp(-self.scale),
                        p * tf.exp(self.scale) - self._shift_model(q))

class Chain(tf.keras.Model):
    """A chain of bijectors"""
    def __init__(self, bijectors):
        super(Chain, self).__init__()
        self.bijectors = bijectors

    def call(self, x):
        for bijector in self.bijectors:
            x = bijector(x)
        return x

    def inverse(self, x):
        for bijector in reversed(self.bijectors):
            x = bijector.inverse(x)
        return x

# Neural networks: standard neural networks that implement arbitrary functions
# used in the bijectors.
class MLP(tf.keras.Model):
    def __init__(self, activation=tf.tanh, return_gradient=True):
        """
        If return_gradient, returns grad(F, x), where F(x) is a scalar
        computed by an MLP and which corresponds to a generating
        function of the shift. Otherwise, just MLP with output same dim
        as input.

        If return_gradient, choose an activation function that is not piecewise
        linear, otherwise the taking gradient kills the x-dependence and bias.
        """
        super(MLP, self).__init__()
        self.return_gradient = return_gradient
        d = 512
        self.dense1 = tf.keras.layers.Dense(d, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(d, activation=activation)

    def build(self, input_shape):
        # call_strategy factory
        if self.return_gradient:
            # No bias in the last layer since we take the gradient of the output
            self.dense3 = tf.keras.layers.Dense(1, activation=None,
                                                use_bias=False)
            self.call_strategy = self.call_with_gradient
        else:
            self.phase_space_dim = input_shape[1]
            self.internal_dim = input_shape[2]
            self.dense3 = tf.keras.layers.Dense(self.phase_space_dim*self.internal_dim,
                                                activation=None)
            self.call_strategy = self.call_without_gradient

    def call_with_gradient(self, x):
        with tf.GradientTape() as g:
            g.watch(x)
            F = self.dense3( self.dense2( self.dense1(x) ) )
        return g.gradient(F, x)

    def call_without_gradient(self, x):
        #x = tf.keras.layers.Flatten(x)  does not work?
        x = tf.reshape(x, shape=(-1, self.phase_space_dim*self.internal_dim))
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return tf.reshape(x, shape=(-1, self.phase_space_dim, self.internal_dim))

    def call(self, x):
        return self.call_strategy(x)

class CNNShiftModel(tf.keras.Model):
    def __init__(self):
        super(CNNShiftModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(32, 2, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv1D(64, 2, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dense = tf.keras.layers.Dense(1, activation=None, use_bias=False)

    def build(self, input_shape):
        # Set the number of channels equals to the input ones
        self.conv_last = tf.keras.layers.Conv1D(int(input_shape[2]), 2, padding='same')

    def call(self, x):
        with tf.GradientTape() as g:
            g.watch(x)
            F = tf.nn.relu(self.bn1(self.conv1(x)))
            F = tf.nn.relu(self.bn2(self.conv2(F)))
            F = tf.nn.relu(self.conv_last(F))
            # 1 residual branch
            F = self.dense(x + F)
        return g.gradient(F, x)

class CNNShiftModel2(tf.keras.Model):
    def __init__(self):
        super(CNNShiftModel2, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(32, 2, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv1D(64, 2, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.GlobalAveragePooling1D()

    def build(self, input_shape):
        # Set the number of channels equals to the input ones. In practice 1.
        self.conv_last = tf.keras.layers.Conv1D(int(input_shape[2]), 2, padding='same')

    def call(self, x):
        with tf.GradientTape() as g:
            g.watch(x)
            F = tf.nn.relu(self.bn1(self.conv1(x)))
            F = tf.nn.relu(self.bn2(self.conv2(F)))
            F = tf.nn.relu(self.conv_last(F))
            # 1 residual branch
#            F = self.pool(x + F)
            F = self.pool(F)
        return g.gradient(F, x)

# Architectures
class MultiScaleArchitecture(tf.keras.Model):
    """Multi scale architecture inspired by realNVP & Glow.
    Here in 1d, using canonical bijectors.
    Extract at each level a subset (half) of the current q,p's.
    """
    def __init__(self, flows):
        """flows: a list of flows to be performed at each scale"""
        super(MultiScaleArchitecture, self).__init__()

        self.flows = flows
        self.num_levels = len(flows)

    def call(self, z):
        assert z.shape[1] % 2**self.num_levels == 0, "z.shape = {}".format(z.shape)
        # TODO: avoid allocating z_list
        z_list = []
        for i in range(self.num_levels-1):
            tmp, z = split(z)
            z_list.append(tmp)
        z_list.append(z)

        x = tf.constant([])
        for i in range(self.num_levels):
            x = safe_concat(z_list[self.num_levels-i-1], x, axis=1)
            x = self.flows[self.num_levels-i-1](x)
        return x

    def inverse(self, x):
        """Inverse function: x -> z.
        Here in 1d, using canonical bijectors.
        """
        assert x.shape[1] % 2**self.num_levels == 0, "x.shape = {}".format(x.shape)
        z = tf.constant([]) # output
        for i in range(self.num_levels-1):
            x = self.flows[i].inverse(x)
            zi, x = split(x)
            z = safe_concat(z, zi, axis=1)
        x = self.flows[-1].inverse(x)
        z = safe_concat(z, x, axis=1)
        return z
