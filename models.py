"""
Keras models to be used in eager execution
"""

import numpy as np
import tensorflow as tf
tfe = tf.contrib.eager

from utils import extract_q_p, join_q_p, int_shape
import utils
from functools import partial
from abc import ABC, abstractmethod
from tensorflow.python.ops.parallel_for import gradients as tf_gradients_ops

DTYPE = tf.float32

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
        return tf.zeros(tf.shape(z)[0], dtype=z.dtype)

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
    def __init__(self, shift=True):
        """ConstantShiftAndScale - general bijector:
        z -> e^s z + t. Can be used to enrich a base distribution as part of
        the base distribution sampling.
        """
        super(ConstantShiftAndScale, self).__init__()
        self._do_shift = shift

    def build(self, sz):
        dims = sz[1:]
        if self._do_shift:
            self.log_scale = tfe.Variable(tf.zeros(dims), name="log_scale")
            self.shift = tfe.Variable(tf.zeros(dims), name="shift")
            self._forward = self._shift_and_scale_forward
            self._inverse = self._shift_and_scale_inverse
        else:
            self.log_scale = tfe.Variable(tf.zeros(dims), name="log_scale")
            self._forward = self._scale_forward
            self._inverse = self._scale_inverse

    def call(self, z):
        return self._forward(z)

    def inverse(self, x):
        return self._inverse(x)

    def log_jacobian_det(self, z):
        return tf.reduce_sum(self.log_scale) * tf.ones(tf.shape(z)[0])

    def _shift_and_scale_forward(self, z):
        return tf.multiply(tf.exp(self.log_scale), z) + self.shift

    def _shift_and_scale_inverse(self, x):
        return tf.multiply(tf.exp(-self.log_scale), x - self.shift)

    def _scale_forward(self, z):
        return tf.multiply(tf.exp(self.log_scale), z)

    def _scale_inverse(self, x):
        return tf.multiply(tf.exp(-self.log_scale), x)

class AffineCoupling(NormalizingFlow):
    def __init__(self, shift_and_scale_model, split_dim, split_sizes, is_positive_shift=False):
        """AffineCoupling (Real NVP)"""
        super(AffineCoupling, self).__init__()
        self._nn = shift_and_scale_model
        self._split_dim = split_dim
        self._split_sizes = split_sizes # [na, nb]
        self._is_positive_shift = is_positive_shift

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
        if self._is_positive_shift:
            shift = tf.abs(shift)
        return log_scale, shift

class AdditiveCoupling(ZeroLogJacobianDetMixin, NormalizingFlow):
    def __init__(self, shift_model, split_dim, split_sizes, is_positive_shift=False):
        """AdditiveCoupling (NICE)"""
        super(AdditiveCoupling, self).__init__()
        self._nn = shift_model
        self._split_dim = split_dim
        self._split_sizes = split_sizes # [na, nb]
        self._is_positive_shift = is_positive_shift

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
        if self._is_positive_shift:
            return tf.abs(self._nn(zb))
        else:
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

class LinearSymplecticTwoByTwo(SymplecticFlow):
    def __init__(self, rand_init=False):
        super(LinearSymplecticTwoByTwo, self).__init__()
        self.rand_init = rand_init

    def build(self, input_size):
        """Initialize to identity or random"""
        input_size = input_size.as_list()
        dof = input_size[1] * input_size[2]
        s_shape = [dof, 1, 1]
        if self.rand_init:
            s1_init = tf.keras.initializers.glorot_normal()(s_shape)
            s2_init = tf.keras.initializers.glorot_normal()(s_shape)
            s3_init = tf.keras.initializers.glorot_normal()(s_shape)
        else:
            s1_init = tf.ones(s_shape)
            s2_init = s3_init = tf.zeros(s_shape)
        s1 = tfe.Variable(s1_init, name="s1")
        s2 = tfe.Variable(s2_init, name="s2")
        s3 = tfe.Variable(s3_init, name="s3")
        s4 = (1 + s2 * s3) / s1
        self.S = tf.concat([tf.concat([s1, s2], axis=2), tf.concat([s3, s4], axis=2)], axis=1)
        self.inverse_S = tf.concat([tf.concat([s4, -s2], axis=2), tf.concat([-s3, s1], axis=2)], axis=1)

    def call(self, x):
        x_shape = tf.shape(x)
        # x_shape = [N,d,n,2] where q=[:,:,:,0], p=[:,:,:,1]
        x = tf.reshape(x, [x_shape[0], x_shape[1]*x_shape[2], 2])
        res = tf.einsum('abc,dac->dab', self.S, x)
        return tf.reshape(res, shape=x_shape)

    def inverse(self, x):
        x_shape = tf.shape(x)
        # x_shape = [N,d,n,2] where q=[:,:,:,0], p=[:,:,:,1]
        x = tf.reshape(x, [x_shape[0], x_shape[1]*x_shape[2], 2])
        res = tf.einsum('abc,dac->dab', self.inverse_S, x)
        return tf.reshape(res, shape=x_shape)

# TODO: It cannot represent the identity since L neq 0 so it leads to nans
# when trying to learn identity.
class LinearSymplectic(SymplecticFlow):
    def __init__(self, squeezing_factors=[1, 1]):
        """q,p -> conv2d( squeeze(q,p) , filters = 1x1, weights=S ) ->
        unsqueeze S symplectic. The squeezing factors chooses the
        dimensionality of the reduced phase space on which S acts (=2*f1*f2)
        """
        super(LinearSymplectic, self).__init__()
        self.f1 = squeezing_factors[0]
        self.f2 = squeezing_factors[1]
        tot_squeezing_factor = self.f1 * self.f2
        c = 2*tot_squeezing_factor # channel dim
        # init S = filters as free symplectic matrix, Sinv.
        # S = [[LQ, L], [PLQ - L^{-1^T}, PL]],
        # with P,Q symmetric, L invertible.
        # TODO: improvide parametrization P,Q
        sz = [tot_squeezing_factor, tot_squeezing_factor]
        init = tf.keras.initializers.orthogonal()
        L = tfe.Variable(init(sz), name="L")
        P = tfe.Variable(init(sz), name="P")
        P = 0.5 * (P + tf.transpose(P))
        Q = tfe.Variable(init(sz), name="Q")
        Q = 0.5 * (Q + tf.transpose(Q))
        A = tf.matmul(L,Q)
        B = L
        C = tf.matmul(tf.matmul(P,L),Q) - tf.matrix_inverse(L, adjoint=True)
        D = tf.matmul(P,L)
        self.S = tf.stack([tf.stack([A,B], axis=1),
                           tf.stack([C,D], axis=1)], axis=0 )
        self.S = tf.reshape(self.S, [1,1,c,c])
        self.Sinv = tf.stack([tf.stack([tf.transpose(D),-tf.transpose(B)],axis=1),
                              tf.stack([-tf.transpose(C),tf.transpose(A)],axis=1)],
                              axis=0 )
        self.Sinv = tf.reshape(self.Sinv, [1,1,c,c])

    def call(self, x):
        x = self.squeeze(x)
        x = tf.nn.conv2d(x, self.S, [1,1,1,1], 'SAME')
        return self.unsqueeze(x)

    def inverse(self, x):
        x = self.squeeze(x)
        x = tf.nn.conv2d(x, self.Sinv, [1,1,1,1], 'SAME')
        return self.unsqueeze(x)

    def squeeze(self, x):
        # x.shape = [N, d, n, 2]
        N,d,n,_ = int_shape(x)
        return tf.reshape(x, [N,
                              d//self.f1,
                              n//self.f2,
                              2*self.f1*self.f2,
                              ])

    def unsqueeze(self, x):
        # x.shape = [N, d/f1, n/f2, 2*f1*f2]
        N,d_over_f1,n_over_f2,_ = int_shape(x)
        return tf.reshape(x, [N,
                              d_over_f1 * self.f1,
                              n_over_f2 * self.f2,
                              2
                              ])

# Hamiltonian vector field
def hamiltonian_vector_field(hamiltonian, x, t, backward_time=False):
    """X_H appearing in Hamilton's eqs: xdot = X_H(x)"""
    # note, t unused.
    dx = tf.gradients(hamiltonian(x),x)[0]
    dq,dp = extract_q_p(dx)
    if not backward_time:
        return join_q_p(dp,-dq)
    else:
        return join_q_p(-dp,dq)

class HamiltonianFlow(SymplecticFlow):
    def __init__(self, hamiltonian, initial_t=0., final_t=10., num_steps=1000):
        """q,p \equiv q(0),p(0) -> q(t),p(t) under Hamilton's eom.

        V1: backpropagate through tf.contrib.odeint
        """
        super(HamiltonianFlow, self).__init__()
        self._t = tf.linspace(initial_t, final_t, num=num_steps)
        # Save the Hamiltonian since it can contain trainable variables
        # accessible from the HamiltonianFlow model.
        self._hamiltonian = hamiltonian
        # Set integrator.
        # TODO: replace with symplectic integrator (leapfrog)
        self._integrate = tf.contrib.integrate.odeint

    def call(self, x0, return_full_state=False):
        """integrate t0 -> tf, x0 is initial condition.
        If not return_full_state, returns the value at tf."""
        hamilton_eqs = partial(hamiltonian_vector_field, self._hamiltonian)
        x = self._integrate(hamilton_eqs, x0, self._t)
        if return_full_state:
            return x
        else:
            return x[-1, ...]

    def inverse(self, x0):
        """integrate tf -> t0, x0 is initial condition. Returns value at t0"""
        hamilton_eqs = partial(hamiltonian_vector_field, self._hamiltonian, backward_time=True)
        x = self._integrate(hamilton_eqs, x0, self._t)
        return x[-1, ...]

class NonLinearSqueezing(SymplecticFlow):
    def __init__(self, f):
        """(q,p) -> (f(q), Df(q)^{-1} p)"""
        super(NonLinearSqueezing, self).__init__()
        #assert isinstance(f, NormalizingFlow), "f needs to have an inverse"
        self._f = f

    def call(self, x):
        q, p = extract_q_p(x)
        q_prime = self._f(q)
        # Df(q)^{-1} = D(f^{-1}( q_prime ))
        df_inverse = tf_gradients_ops.jacobian( self._f.inverse(q_prime), q_prime,use_pfor=True )
        return join_q_p(q_prime, tf.tensordot(df_inverse, p, [[4,5,6,7],[0,1,2,3]]))

    def inverse(self, z):
        q, p = extract_q_p(x)
        q_prime = self._f.inverse(q)
        df = tf_gradients_ops.jacobian( self._f(q_prime), q_prime,use_pfor=True )
        return join_q_p(q_prime, tf.tensordot(df, p, [[4,5,6,7],[0,1,2,3]]))

# class ActNorm(SymplecticFlow):
#     def __init__(self):
#         """At init, shifts activations to have zero center.
#         Then, learnable shift."""
#         super(ActNorm, self).__init__()
#         self._shift = tfe.Variable(tf.zeros([2]), name="shift")
#         self._called = False
#
#     def call(self, x):
#         if self._called == False:
#             # First time called, init to activations mean per channel
#             self._shift = tf.reduce_mean(x, [0,1,2])
#             self._called = True
#         return x - self._shift
#
#     def inverse(self, z):
#         return z + self._shift

class ZeroCenter(SymplecticFlow):
    def __init__(self, decay=0.99, debias=False, is_training=True):
        """Shifts activations to have zero center. Add learnable offset.
        call (forward) used during training, normalizes:
        y = x - mean(x) + offset
        while inverse de-normalizes:
        y = x + mean(x) - offset

        If training flag, compute running avg mean during call and use batch
        mean to zero-center. When training false, use the moving mean to
        zero-center.
        Inverse method used for inference phase only, so assumes training=False.
        Therefore, while tranining, call() and inverse() are not inverse of each
        other.

        For a rough implementation, see:
        https://gluon.mxnet.io/chapter04_convolutional-neural-networks/cnn-batch-norm-scratch.html
        """
        super(ZeroCenter, self).__init__()
        self._decay = decay
        if debias:
            raise NotImplementedError
        # TODO: update debias
        # self._debias = debias
        # if self._debias:
        #     self._num_updates = tfe.Variable(0,
        #                                      name="num_updates",
        #                                      dtype = tf.int64,
        #                                      trainable=False)
        self.is_training = is_training

    def build(self, in_sz):
        num_channels = in_sz[-1]
        self._offset = tfe.Variable(tf.zeros([num_channels]), name="offset")
        self._moving_mean = tfe.Variable(tf.zeros([num_channels]), name="mean",
                                         trainable=False)

    def call(self, x):
        if self.is_training:
            minibatch_mean_per_channel = tf.reduce_mean(x, [0,1,2])
            # Need assign to update the variable. This code is not run more than
            # once, but the assign op is added to the graph and run every time.
            train_mean = tf.assign(self._moving_mean,
                self._decay * self._moving_mean + \
                (1.-self._decay) * minibatch_mean_per_channel)

            # TODO: update debias
            # if self._debias:
            #     # Debias: divide by geometric sum, see Adam paper sec 3.
            #     self._num_updates = self._num_updates + 1
            #     self._moving_mean = self._moving_mean / \
            #         (1.-self._decay ** tf.cast(self._num_updates, DTYPE))

            # This runs the with block after the ops it depends on, so that
            # moving_mean is updated every iteration.
            with tf.control_dependencies([train_mean]):
                return x - minibatch_mean_per_channel + self._offset
        else:
            return x - self._moving_mean + self._offset

    def inverse(self, z):
        assert not self.is_training
        return z + self._moving_mean - self._offset

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
        linear, otherwise taking the gradient kills the x-dependence and bias.
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
            self.dense3 = tf.keras.layers.Dense(1,use_bias=False)
            self.call_strategy = self.call_symplectic_shift
        elif self.mode == "shift":
            self.in_space_dim = np.prod(input_shape[1:])
            if self.out_shape == None:
                self.out_shape = [-1] + input_shape[1:].as_list()
            dim = np.prod(self.out_shape[1:])
            self.dense3 = tf.keras.layers.Dense(dim)
            self.call_strategy = self.call_shift
        elif self.mode == "shift_and_scale":
            self.in_space_dim = np.prod(input_shape[1:])
            if self.out_shape == None:
                self.out_shape = [-1] + input_shape[1:].as_list()
                # Add to the channel
                self.out_shape[-1] = 2 * self.out_shape[-1]
            dim = np.prod(self.out_shape[1:])
            self.dense3 = tf.keras.layers.Dense(dim)
            self.call_strategy = self.call_shift_and_scale
        else:
            assert False, "Wrong mode"

    def forward(self, x):
        return self.dense3( self.dense2( self.dense1(tf.layers.flatten(x)) ) )

    def call_symplectic_shift(self, x):
        def f(x):
            return self.forward(x)
        return utils.grad(f, [x])[0]

    def call_shift(self, x):
        x = self.forward(x)
        return tf.reshape(x, shape=self.out_shape)

    def call_shift_and_scale(self, x):
        x = self.forward(x)
        return tf.reshape(x, shape=self.out_shape)

    def call(self, x):
        return self.call_strategy(x)

class IrrotationalMLP(tf.keras.Model):
    """NN for irrotational vector field. Impose the symmetry at the level of weights of MLP."""
    def __init__(self, activation=tf.nn.tanh, width=512, rand_init=True):
        super(IrrotationalMLP, self).__init__()
        self.width = width
        self.rand_init = rand_init
        self.act = activation

    def build(self, input_shape):
        input_shape = input_shape.as_list()
        input_dimension = np.prod(input_shape[1:])
        shape = (self.width,)
        if self.rand_init:
            W2_init = tf.keras.initializers.glorot_normal()(shape)
            b2_init = tf.keras.initializers.glorot_normal()(shape)
        else:
            W2_init = tf.zeros(shape)
            b2_init = tf.zeros(shape)
        self.W1 = tfe.Variable(tf.keras.initializers.glorot_normal()((input_dimension, self.width)), name="W1")
        self.W2 = tfe.Variable(W2_init, name="W2")
        self.b1 = tfe.Variable(tf.keras.initializers.glorot_normal()(shape), name="b1")
        self.b2 = tfe.Variable(b2_init, name="b2")
        self.b3 = tfe.Variable(tf.zeros((input_dimension,)), name="b3")

    def call(self, x):
        x_shape = tf.shape(x)
        x = tf.layers.flatten(x)
        x = self.act(tf.matmul(x, self.W1) + self.b1) # x.shape = (batch, width)
        x = self.act(tf.multiply(self.W2, x) + self.b2) # x.shape = (batch, width)
        x = tf.matmul(x, self.W1, transpose_b=True) + self.b3  # x.shape = (batch, input_dimension)
        return tf.reshape(x, x_shape)

class MLPHamiltonian(tf.keras.Model):
    def __init__(self, d=512, activation=tf.nn.softplus):
        """A neural network with scalar output. Arbitrary input x"""
        super(MLPHamiltonian, self).__init__()
        self.dense1 = tf.keras.layers.Dense(d,
                                            activation=activation,
                                            kernel_initializer=tf.keras.initializers.Orthogonal())
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(d,
                                            activation=activation,
                                            kernel_initializer=tf.keras.initializers.Orthogonal())
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, x):
        x = tf.layers.flatten(x)
        x = self.bn1( self.dense1(x) )
        x = self.bn2( self.dense2(x) )
        x = self.dense3(x)
        return x

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

    def inverse(self, x, stop_at=0):
        """stop_at is the bijector to stop at. Default = 0 means till end."""
        for bijector in reversed(self.bijectors[stop_at:]):
            x = bijector.inverse(x)
        return x

    def log_jacobian_det(self, z):
        ldj = tf.zeros(tf.shape(z)[0], dtype=z.dtype)
        for bijector in self.bijectors[:-1]:
            ldj += bijector.log_jacobian_det(z)
            z = bijector(z)
        return ldj + self.bijectors[-1].log_jacobian_det(z)

    def set_is_training(self, tf):
        """Set is_training attribute of bijectors."""
        for i, bijector in enumerate(self.bijectors):
            if hasattr(bijector, 'is_training'):
                self.bijectors[i].is_training = tf

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
        ndims = len(z.get_shape().as_list())
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
    def __init__(self, first_only=False):
        super(OscillatorFlow, self).__init__()
        self._first_only = first_only

    def call(self, z):
        phi, I = extract_q_p(z)
        if self._first_only:
            sqrt_two_I = tf.sqrt(2. * I[:,0,0,0])
            q000 = tf.multiply(sqrt_two_I, tf.sin(phi[:,0,0,0]))
            p000 = tf.multiply(sqrt_two_I, tf.cos(phi[:,0,0,0]))
            p = I
            q = phi
            p = self._assign_000(p, p000)
            q = self._assign_000(q, q000)
        else:
            sqrt_two_I = tf.sqrt(2. * I)
            q = tf.multiply(sqrt_two_I, tf.sin(phi))
            p = tf.multiply(sqrt_two_I, tf.cos(phi))
        return join_q_p(q,p)

    def inverse(self, x):
        qq, pp = extract_q_p(x)
        if self._first_only:
            I000 = 0.5 * (tf.square(qq[:,0,0,0]) + tf.square(pp[:,0,0,0]))
            phi000 = tf.atan(qq[:,0,0,0] / pp[:,0,0,0])
            I = pp
            phi = qq
            I = self._assign_000(I, I000)
            phi = self._assign_000(phi, phi000)
        else:
            phi = tf.atan(qq / pp)
            I = 0.5 * (tf.square(qq) + tf.square(pp))
        return join_q_p(phi, I)

    def _assign_000(self, v, v000):
        # v is of shape [N,d,n,1]
        d = tf.shape(v)[1]
        n = tf.shape(v)[2]
        v = tf.reshape(v, [-1,d*n,1])
        v000 = tf.reshape(v000, [-1,1,1])
        v = tf.concat([v000, v[:,1:,:]], 1)
        return tf.reshape(v, [-1,d,n,1])
