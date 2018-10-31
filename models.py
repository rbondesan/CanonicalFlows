"""
Keras models to be used in eager execution
"""

import numpy as np
import tensorflow as tf
tfe = tf.contrib.eager

from utils import extract_q_p, join_q_p
from abc import ABC, abstractmethod

# Interfaces and mixins
class NormalizingFlow(ABC, tf.keras.Model):
    """Interface for keras models representing normalizing flows."""
    def __init__(self):
        super(NormalizingFlow, self).__init__()

    @abstractmethod
    def inverse(self, x):
        """The inverse of self.call"""
        pass

    @abstractmethod
    def log_jacobian_det(self, z):
        """Log Jacobian Determinant. If z is of shape
        (N, m1, m2, m3, ..., c), the return value is of shape (N,)."""
        pass

class ZeroLogJacobianDetMixin():
    """Mixin implementating zero log Jacobian determinant."""
    def log_jacobian_det(self, z):
        return tf.zeros(z.shape[0], dtype=z.dtype)

class SymplecticFlow(ZeroLogJacobianDetMixin, NormalizingFlow):
    """Interface for symplectic flows. Implement only call and inverse.
    Understood that for phase space variables only and the flow is symplectic."""
    def __init__(self):
        super(SymplecticFlow, self).__init__()

# Flows (or bijectors)
class AngleInvariantFlow(NormalizingFlow):
    """Wrapper of a generic normalizing flow which acts only on actions and
    leaves the angles invariant."""
    def __init__(self, flow):
        """Here flow is a generic flow"""
        super(AngleInvariantFlow, self).__init__()
        assert isinstance(flow, NormalizingFlow), "flow needs to be a NormalizingFlow"
        self._flow = flow

    def call(self, z):
        # Chose p as the action, q as the angle.
        q,p = extract_q_p(z)
        return join_q_p(q, self._flow(p))

    def inverse(self, x):
        # Chose p as the action, q as the angle.
        q,p = extract_q_p(x)
        return join_q_p(q, self._flow.inverse(p))

    def log_jacobian_det(self, z):
        # Chose p as the action, q as the angle.
        q,p = extract_q_p(z)
        return self._flow.log_jacobian_det(p)

class IdFlow(ZeroLogJacobianDetMixin, NormalizingFlow):
    def __init__(self):
        """Identity flow - generic flow"""
        super(IdFlow, self).__init__()

    def call(self, z):
        return z

    def inverse(self, x):
        return x

# TODO: test
class Permute(ZeroLogJacobianDetMixin, NormalizingFlow):
    def __init__(self, split_dim, split_sizes):
        """Permute - generic flow"""
        super(Permute, self).__init__()
        self.split_dim = split_dim
        self.split_sizes = split_sizes

    def call(self, z):
        z1,z2 = tf.split(z, self.split_sizes, self.split_dim)
        return tf.concat([z2,z1], self.split_dim)

    def inverse(self, x):
        return self.call(x)

class ConstantShiftAndScale(NormalizingFlow):
    def __init__(self):
        """ConstantShiftAndScale - general bijector:
        z -> e^s z + t"""
        super(ConstantShiftAndScale, self).__init__()

    def build(self, sz):
        dims = sz[1:]
        self.log_scale = tfe.Variable(tf.random_normal(dims), name="log_scale")
        self.shift = tfe.Variable(tf.random_normal(dims), name="shift")

    def call(self, z):
        return tf.multiply(tf.exp(self.log_scale), z) + self.shift

    def inverse(self, x):
        return tf.multiply(tf.exp(-self.log_scale), z - self.shift)

    def log_jacobian_det(self, z):
        return tf.reduce_sum(self.log_scale) * tf.ones(z.shape[0])

class AffineCoupling(NormalizingFlow):
    def __init__(self, shift_and_scale_model, split_dim, split_sizes):
        """AffineCoupling (Real NVP)"""
        super(AffineCoupling, self).__init__()
        self._nn = shift_and_scale_model
        self._split_dim = split_dim
        self._split_sizes = split_sizes # [na, nb]

    def call(self, z):
        za,zb = self._split_a_b(z)
        log_scale, shift = self._compute_log_scale_shift(zb)
        return self._join_a_b( tf.multiply(tf.exp(log_scale), za) + shift, zb )

    def inverse(self, x):
        xa,xb = self._split_a_b(x)
        log_scale, shift = self._compute_log_scale_shift(xb)
        return self._join_a_b( tf.multiply(xa - shift, tf.exp(-log_scale)), xb )

    def log_jacobian_det(self, z):
        # Can we remove the call to compute_s_t here? Maybe call returns
        # both output and logDetJ?
        za,zb = self._split_a_b(z)
        log_scale, shift = self._compute_log_scale_shift(zb)
        # Assume z is [N,d,n,c]
        return tf.reduce_sum( log_scale, [1,2,3] )

    # Private methods
    def _split_a_b(self, z):
        return tf.split(z, self._split_sizes, self._split_dim)

    def _join_a_b(self, za, zb):
        return tf.concat([za, zb], self._split_dim)

    def _compute_log_scale_shift(self, zb):
        # Assume _nn puts the extra copies along the channel dim -> extract q,p
        log_scale, shift = extract_q_p(self._nn(zb))
        return log_scale, shift

class AdditiveCoupling(ZeroLogJacobianDetMixin, NormalizingFlow):
    def __init__(self, shift_model, split_dim, split_sizes):
        """AdditiveCoupling (NICE)"""
        super(AdditiveCoupling, self).__init__()
        self._nn = shift_model
        self._split_dim = split_dim
        self._split_sizes = split_sizes # [na, nb]

    def call(self, z):
        za,zb = self._split_a_b(z)
        shift = self._compute_shift(zb)
        return self._join_a_b( za + shift, zb )

    def inverse(self, x):
        xa,xb = self._split_a_b(x)
        shift = self._compute_shift(xb)
        return self._join_a_b( xa - shift, xb )

    # Private methods
    def _split_a_b(self, z):
        return tf.split(z, self._split_sizes, self._split_dim)

    def _join_a_b(self, za, zb):
        return tf.concat([za, zb], self._split_dim)

    def _compute_shift(self, zb):
        # Assume _nn returns an output of size za.shape
        return self._nn(zb)

class SymplecticAdditiveCoupling(SymplecticFlow):
    def __init__(self, shift_model):
        """q,p -> q,p + NN(q).
        AdditiveCoupling (NICE) bijector for phase space variables only."""
        super(SymplecticAdditiveCoupling, self).__init__()
        self._shift_model = shift_model

    def call(self, x):
        q,p = extract_q_p(x)
        return join_q_p(q, p + self._shift_model(q))

    def inverse(self, x):
        q,p = extract_q_p(x)
        return join_q_p(q, p - self._shift_model(q))

class SymplecticExchange(SymplecticFlow):
    """q,p -> p,-q"""
    def __init__(self):
        super(SymplecticExchange, self).__init__()

    def call(self, x):
        q,p = extract_q_p(x)
        return join_q_p(p,-q)

    def inverse(self, x):
        q,p = extract_q_p(x)
        return join_q_p(-p,q)

class SqueezeAndShift(SymplecticFlow):
    def __init__(self, shift_model, scalar_scale=True):
        """q,p -> q * e^s , e^(-s) * (p + shift_model(q)).
        """
        super(SqueezeAndShift, self).__init__()
        self._shift_model = shift_model
        # scale is a scalar for lattice shift covariance.
        self.scalar_scale = scalar_scale

    def build(self, input_size):
        # Initialize scale to zeros
        if self.scalar_scale:
            self.scale_exponent = tfe.Variable(tf.zeros([1]), name="scale")
        else:
            sz = [input_size[1], input_size[2], 1]
            self.scale_exponent = tfe.Variable(tf.zeros(sz), name="scale")

    def call(self, x):
        q,p = extract_q_p(x)
        return join_q_p(q * tf.exp(self.scale_exponent),
                        tf.exp(-self.scale_exponent) * (p + self._shift_model(q)))

    def inverse(self, x):
        q,p = extract_q_p(x)
        return join_q_p(q * tf.exp(-self.scale_exponent),
                        p * tf.exp(self.scale_exponent) - self._shift_model(q))

# Neural networks: standard neural networks that implement arbitrary functions
# used in the bijectors.
class MLP(tf.keras.Model):
    def __init__(self, activation=tf.nn.softplus, mode="symplectic_shift",
                out_shape=None):
        """
        mode:
            - "symplectic_shift": MLP : in_space -> R, return gradient(F, input)
            - "shift": MLP : out_dim = in_dim if out_shape==None
            - "shift_and_scale": MLP : out_dim = 2*in_dim if out_shape==None

        If symplectic_shift, choose an activation function that is not piecewise
        linear, otherwise the taking gradient kills the x-dependence and bias.
        """
        super(MLP, self).__init__()
        self.mode = mode
        d = 512
        self.dense1 = tf.keras.layers.Dense(d, activation=activation)
        self.dense2 = tf.keras.layers.Dense(d, activation=activation)
        self.out_shape = out_shape

    def build(self, input_shape):
        # Set call_strategy factory
        if self.mode == "symplectic_shift":
            # No bias in the last layer since we take the gradient of the output
            self.dense3 = tf.keras.layers.Dense(1, activation=None,
                                                use_bias=False)
            self.call_strategy = self.call_symplectic_shift
        elif self.mode == "shift":
            self.in_space_dim = tf.reduce_prod(input_shape[1:])
            if self.out_shape == None:
                self.out_shape = [-1] + input_shape[1:].as_list()
            dim = tf.reduce_prod(self.out_shape[1:])
            self.dense3 = tf.keras.layers.Dense(dim, activation=None)
            self.call_strategy = self.call_shift
        elif self.mode == "shift_and_scale":
            self.in_space_dim = tf.reduce_prod(input_shape[1:])
            if self.out_shape == None:
                self.out_shape = [-1] + input_shape[1:].as_list()
                # Add to the channel
                self.out_shape[-1] = 2 * self.out_shape[-1]
            dim = tf.reduce_prod(self.out_shape[1:])
            self.dense3 = tf.keras.layers.Dense(dim, activation=None)
            self.call_strategy = self.call_shift_and_scale
        else:
            assert False, "Wrong mode"

    def forward(self, x):
        return self.dense3( self.dense2( self.dense1(tf.layers.flatten(x)) ) )

    def call_symplectic_shift(self, x):
        with tf.GradientTape() as g:
            g.watch(x)
            F = self.forward(x)
        return g.gradient(F, x)

    def call_shift(self, x):
        x_shape = x.shape
        x = self.forward(x)
        return tf.reshape(x, shape=x_shape)

    def call_shift_and_scale(self, x):
        x = self.forward(x)
        return tf.reshape(x, shape=self.out_shape)

    def call(self, x):
        return self.call_strategy(x)

# TODO: update
# class CNNShiftModel(tf.keras.Model):
#     def __init__(self):
#         super(CNNShiftModel, self).__init__()
#         self.conv1 = tf.keras.layers.Conv1D(32, 2, padding='same')
#         self.bn1 = tf.keras.layers.BatchNormalization()
#         self.conv2 = tf.keras.layers.Conv1D(64, 2, padding='same')
#         self.bn2 = tf.keras.layers.BatchNormalization()
#         self.dense = tf.keras.layers.Dense(1, activation=None, use_bias=False)
#
#     def build(self, input_shape):
#         # Set the number of channels equals to the input ones
#         self.conv_last = tf.keras.layers.Conv1D(int(input_shape[2]), 2, padding='same')
#
#     def call(self, x):
#         with tf.GradientTape() as g:
#             g.watch(x)
#             F = tf.nn.relu(self.bn1(self.conv1(x)))
#             F = tf.nn.relu(self.bn2(self.conv2(F)))
#             F = tf.nn.relu(self.conv_last(F))
#             # 1 residual branch
#             F = self.dense(x + F)
#         return g.gradient(F, x)
#
# class CNNShiftModel2(tf.keras.Model):
#     def __init__(self):
#         super(CNNShiftModel2, self).__init__()
#         self.conv1 = tf.keras.layers.Conv1D(32, 2, padding='same')
#         self.bn1 = tf.keras.layers.BatchNormalization()
#         self.conv2 = tf.keras.layers.Conv1D(64, 2, padding='same')
#         self.bn2 = tf.keras.layers.BatchNormalization()
#         self.pool = tf.keras.layers.GlobalAveragePooling1D()
#
#     def build(self, input_shape):
#         # Set the number of channels equals to the input ones. In practice 1.
#         self.conv_last = tf.keras.layers.Conv1D(int(input_shape[2]), 2, padding='same')
#
#     def call(self, x):
#         with tf.GradientTape() as g:
#             g.watch(x)
#             F = tf.nn.softplus(self.bn1(self.conv1(x)))
#             F = tf.nn.softplus(self.bn2(self.conv2(F)))
#             F = tf.nn.softplus(self.conv_last(F))
#             # 1 residual branch
# #            F = self.pool(x + F)
#             F = self.pool(F)
#         return g.gradient(F, x)

# Architectures
class Chain(NormalizingFlow):
    """A chain of bijectors"""
    def __init__(self, bijectors):
        super(Chain, self).__init__()
        self.bijectors = bijectors

    def call(self, z):
        for bijector in self.bijectors:
            z = bijector(z)
        return z

    def inverse(self, x):
        for bijector in reversed(self.bijectors):
            x = bijector.inverse(x)
        return x

    def log_jacobian_det(self, z):
        ldj = tf.zeros(z.shape[0], dtype=z.dtype)
        for bijector in self.bijectors[:-1]:
            ldj += bijector.log_jacobian_det(z)
            z = bijector(z)
        return ldj + self.bijectors[-1].log_jacobian_det(z)

# TODO: update
# class MultiScaleArchitecture(tf.keras.Model):
#     """Multi scale architecture inspired by realNVP & Glow.
#     Here in 1d, using canonical bijectors.
#     Extract at each level a subset (half) of the current q,p's.
#     """
#     def __init__(self, flows):
#         """flows: a list of flows to be performed at each scale"""
#         super(MultiScaleArchitecture, self).__init__()
#
#         self.flows = flows
#         self.num_levels = len(flows)
#
#     def call(self, z):
#         assert z.shape[1] % 2**self.num_levels == 0, "z.shape = {}".format(z.shape)
#         # TODO: avoid allocating z_list
#         z_list = []
#         for i in range(self.num_levels-1):
#             tmp, z = split(z)
#             z_list.append(tmp)
#         z_list.append(z)
#
#         x = tf.constant([])
#         for i in range(self.num_levels):
#             x = safe_concat(z_list[self.num_levels-i-1], x, axis=1)
#             x = self.flows[self.num_levels-i-1](x)
#         return x
#
#     def inverse(self, x):
#         """Inverse function: x -> z.
#         Here in 1d, using canonical bijectors.
#         """
#         assert x.shape[1] % 2**self.num_levels == 0, "x.shape = {}".format(x.shape)
#         z = tf.constant([]) # output
#         for i in range(self.num_levels-1):
#             x = self.flows[i].inverse(x)
#             zi, x = split(x)
#             z = safe_concat(z, zi, axis=1)
#         x = self.flows[-1].inverse(x)
#         z = safe_concat(z, x, axis=1)
#         return z


# Classes used in the test - to be moved to the test file
class TimesTwoBijector(NormalizingFlow):
    """z -> 2*z"""
    def __init__(self):
        super(TimesTwoBijector, self).__init__()

    def call(self, z):
        return 2.*z

    def inverse(self, x):
        return x/2.

    def log_jacobian_det(self, z):
        # Log Det(J) = sum ( Log (2 * [1,1,...,1] ) ), with [1,1,...,1] of shape z.shape(1:-1)
        # so log(2) * n, where n = prod(z.shape(1:-1))
        #val = tf.log(2.) * tf.cast(tf.reduce_prod(z.shape[1:-1]), dtype=z.dtype)
        #return tf.constant(val, shape=z.shape[0])
        ndims = len(z.shape.as_list())
        return tf.reduce_sum(tf.log(2 * tf.ones_like(z)), list(range(1,ndims)))

# TODO: -> NormalizingFlow
# class TimesBijector(tf.keras.Model):
#     """q,p -> 2*q,2*p"""
#     def __init__(self, multiplier):
#         super(TimesBijector, self).__init__()
#         self.multiplier = multiplier

#     def call(self, x):
#         return self.multiplier*x

#     def inverse(self, x):
#         return x/self.multiplier

class ReplicateAlongChannels(tf.keras.Model):
    """z -> x = concat([z,z], -1)"""
    def __init__(self, truncate=False):
        super(ReplicateAlongChannels, self).__init__()
        self.truncate = truncate

    def call(self, z):
        if self.truncate == False:
            return tf.concat([z,z], -1)
        else:
            z = z[:,:1,:,:]
            return tf.concat([z,z], -1)

class ConstantNN(tf.keras.Model):
    def __init__(self,  out_shape, val):
        super(ConstantNN, self).__init__()
        self.out_shape = out_shape
        self.val = val

    def call(self, z):
        return self.val * tf.ones(self.out_shape, dtype=z.dtype)

class SinPhiFlow(SymplecticFlow):
    def __init__(self):
        super(SinPhiFlow, self).__init__()
        # Small number to avoid division by zero
        self.eps = 0.00001

    def call(self, z):
        phi, I = extract_q_p(z)
        q = tf.sin(phi)
        p = tf.multiply(I, 1/(tf.cos(phi) + self.eps))
        return join_q_p(q,p)

    def inverse(self, x):
        q, p = extract_q_p(x)
        phi = tf.asin(q)
        I = tf.multiply(p, tf.cos(phi) + self.eps)
        return join_q_p(phi, I)

class OscillatorFlow(SymplecticFlow):
    """Map to symplectic polar coordinates. Works for arbitrary (N,d,n,2)
    tensors."""
    def __init__(self):
        super(OscillatorFlow, self).__init__()

    def call(self, z):
        phi, I = extract_q_p(z)
        sqrt_two_I = tf.sqrt(2. * I)
        q = tf.multiply(sqrt_two_I, tf.sin(phi))
        p = tf.multiply(sqrt_two_I, tf.cos(phi))
        return join_q_p(q,p)

    def inverse(self, x):
        qq, pp = extract_q_p(x)
        phi = tf.atan(qq / pp)
        I = 0.5 * (tf.square(qq) + tf.square(pp))
        return join_q_p(phi, I)
