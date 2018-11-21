"""
Utility functions
"""

import functools
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
tfe = tf.contrib.eager
import tensorflow_probability as tfp
tfd = tfp.distributions

DTYPE = tf.float32
NP_DTYPE=np.float32

# Tensor manipulation
def extract_q_p(x):
    """split along the channel axis: q = x[...,0], p = x[...,1]"""
    tf.assert_equal(tf.shape(x)[-1], tf.constant(2))
    # Use :: to keep number of dimensions
    return x[...,::2], x[...,1::2]

def join_q_p(q, p):
    """join q,p along channel axis"""
    tf.assert_equal(tf.shape(q), tf.shape(p))
    tf.assert_equal(tf.shape(q)[-1], tf.constant(1))
    return tf.concat([q, p], -1)

def get_phase_space_dim(sh):
    """sh = (N,n1,n2,...,nd,2) -> n1 * n2 * ... * nd * 2"""
    return tf.reduce_prod(sh[1:])

def int_shape(x):
    if str(x.get_shape()[0]) != '?':
        return list(map(int, x.get_shape()))
    return [-1]+list(map(int, x.get_shape()[1:]))

# TODO: update
# def split(x):
#     """Split into two halves along the phase space dimension, returning z1,z2"""
#     # Assume shape(x) = [batch, phase_space_dim, 1]
#     x_shape = x.get_shape()
#     phase_space_dim = int(x_shape[1])
#     assert x_shape[2] == 1
#     assert phase_space_dim % 2 == 0, "phase_space_dim: {}".format(phase_space_dim)
#     return x[:, :phase_space_dim//2, :], x[:, phase_space_dim//2:, :]

# TODO: update
# def safe_concat(x, y, axis=1):
#     """If x is empty, return y. If y is empty, return x."""
#     if x.shape != 0 and y.shape != 0:
#         return tf.concat([x,y], axis)
#     elif x.shape != 0:
#         return x
#     elif y.shape != 0:
#         return y
#     else:
#         return tf.constant([])

# Distributions
class BaseDistributionActionAngle():
    def __init__(self, settings, action_dist='exponential'):
        sh = [settings['d'], settings['num_particles'], 1]
        # Actions
        if action_dist == 'exponential':
            self.base_dist_u = tfd.Independent(tfd.Exponential(rate=tf.ones(sh, DTYPE)),
                                               reinterpreted_batch_ndims=len(sh))
        elif action_dist == 'normal':
            self.base_dist_u = tfd.MultivariateNormalDiag(loc=tf.zeros(sh, DTYPE))
        # Angles
        self.base_dist_phi = tfd.Independent(tfd.Uniform(low=tf.zeros(sh, DTYPE),
                                                         high=2*np.pi*tf.ones(sh, DTYPE)),
                                             reinterpreted_batch_ndims=len(sh))

    def sample(self, N):
        u = self.base_dist_u.sample(N)
        phi = self.base_dist_phi.sample(N)
        return join_q_p(phi, u)

class BaseDistributionNormal():
    def __init__(self, settings):
        sh = [settings['d'], settings['num_particles'], 2]
        self.base_dist_z = tfd.MultivariateNormalDiag(loc=tf.zeros(sh, DTYPE))

    def sample(self, N):
        return self.base_dist_z.sample(N)

class BaseDistributionIntegralsOfMotion():
    def __init__(self, settings):
        """In the integrals of motion basis and their conjugate (F,psi), we
        can choose F_1 = H, so that the distribution is exponential for the first
        "momentum" variable and uniform for all the others. Here use standard
        normalization as the ranges will be learnt as part of the
        base-distribution-sampler part of the model"""
        # F
        self.base_dist_F1 = tfd.Exponential(rate=1.)
        sh = (settings['d'] * settings['num_particles'] - 1,)
        self.base_dist_otherF = tfd.Independent(tfd.Uniform(low=tf.zeros(sh, DTYPE),
                                                            high=2*np.pi*tf.ones(sh, DTYPE)),
                                                reinterpreted_batch_ndims=len(sh))
        # Psi
        self.sh = [settings['d'], settings['num_particles'], 1]
        self.base_dist_Psi = tfd.Independent(tfd.Uniform(low=tf.zeros(self.sh, DTYPE),
                                                         high=tf.ones(self.sh, DTYPE)),
                                             reinterpreted_batch_ndims=len(self.sh))

    def sample(self, N):
        F1 = tf.reshape(self.base_dist_F1.sample(N), shape=[N,1]) # sh = [N,1]
        otherF = self.base_dist_otherF.sample(N)  # sh = [N,d*n-1]
        F = tf.concat([F1, otherF], 1)       # sh = [N,d*n]
        F = tf.reshape(F, shape=[N]+self.sh) # sh = [N,d,n,1]
        Psi = self.base_dist_Psi.sample(N)
        return join_q_p(Psi, F)

# TODO: update
# Symmetry utils
# def lattice_shift(x):
#     """x = (q_1,p_1,q_2,p_2, ..., q_n,p_n)
#     -> (q_n,p_n,q_1,p_1, ..., q_{n-1},p_{n-1})"""
#     q, p = extract_q_p(x)
#     q_shifted = tf.manip.roll(q, shift=1, axis=1)
#     p_shifted = tf.manip.roll(p, shift=1, axis=1)
#     return join_q_p(q_shifted, p_shifted)

def is_symplectic(model, x):
    """Test if model is simplectic at x.
    Assume x.shape = (1,n1,n2,..,nd,2)"""
    x_shape = x.shape
    assert x_shape[0] == 1 and x_shape[-1] == 2
    phase_space_dim = int(get_phase_space_dim(x_shape))
    x = tf.reshape(x, (phase_space_dim,))
    with tf.GradientTape(persistent=True) as g:
        g.watch(x)
        y = tf.reshape(model(tf.reshape(x,x_shape)), (phase_space_dim,))
        # Need to wrap around a list to compute grad of elements. Why? tf bug?
        ys = [y[i] for i in range(phase_space_dim)]
    # Do the rest in numpy
    jacobian = np.zeros((phase_space_dim, phase_space_dim))
    # TODO: need to rewrite with parallel for
    for i in range(phase_space_dim):
        # set the i-th row
        jacobian[i,:] = np.reshape(g.gradient(ys[i], x).numpy(), (phase_space_dim,))
    del g

    iSigma2 = np.array([[0,1],[-1,0]])
    omega = np.kron(np.eye(phase_space_dim//2), iSigma2)
    omega_tilde = np.dot(np.dot(jacobian, omega), np.transpose(jacobian))
    return np.allclose(omega_tilde, omega, rtol=1e-05, atol=1e-08)

def checkpoint_save(settings, optimizer, model, optimizer_step):
    name = settings['hamiltonian'].__name__
    for key, val in settings.items():
        if key == 'hamiltonian':
            continue
        else:
            name += "_"+key+str(val)
    checkpoint_dir = 'saved_models'
    checkpoint_prefix = os.path.join(checkpoint_dir, name)
    root = tfe.Checkpoint(optimizer=optimizer,
                          model=model,
                          optimizer_step=optimizer_step)
    root.save(file_prefix=checkpoint_prefix)

def checkpoint_restore(settings, model, optimizer=None, optimizer_step=None):
    name = settings['hamiltonian'].__name__
    for key, val in settings.items():
        if key == 'hamiltonian':
            continue
        else:
            name += "_"+key+str(val)
    checkpoint_dir = 'saved_models'
    checkpoint_prefix = os.path.join(checkpoint_dir, name)
    root = tfe.Checkpoint(optimizer=optimizer,
                          model=model,
                          optimizer_step=optimizer_step)
    root.restore(tf.train.latest_checkpoint('saved_models'))

# Test utils
def assert_equal(a,b):
    """a,b are tf.Tensor."""
    assert tf.reduce_all(tf.equal(a,b)), "Not equal: {}, {}".format(a,b)

def assert_allclose(x, y, rtol=1e-5, atol=1e-8):
    assert tf.reduce_all(tf.abs(x - y) <= tf.abs(y) * rtol + atol), "Not equal: {}, {}".format(x,y)

def run_eagerly(func):
    """
    Use eager execution in graph mode.
    See https://stackoverflow.com/questions/50143896/both-eager-and-graph-execution-in-tensorflow-tests
    """
    @functools.wraps(func)
    def eager_fun(*args, **kwargs):
        with tf.Session() as sess:
            sess.run(tfe.py_func(func, inp=list(kwargs.values()), Tout=[]))

    return eager_fun

# TODO: update
# Visualization
def visualize_chain_bijector_1d(model, x, sess=None):
    """Assumes eager mode"""
    if tf.executing_eagerly():
        samples = [x]
    else:
        samples = [sess.run(x)]
    names = ["base_dist"]
    for bijector in model.bijectors:
        x = bijector(x)
        if tf.executing_eagerly():
            samples.append(x)
        else:
            samples.append(sess.run(x))
        names.append(bijector.name)
    f, arr = plt.subplots(1, len(samples), figsize=(4 * (len(samples)), 4))
    if tf.executing_eagerly():
        X0 = tf.reshape(samples[0].numpy(), shape=(samples[0].shape[0], 2))
    else:
        X0 = np.reshape(samples[0], (samples[0].shape[0], 2))
    for i in range(len(samples)):
        if tf.executing_eagerly():
            X1 = tf.reshape(samples[i].numpy(), shape=(samples[0].shape[0], 2))
        else:
            X1 = np.reshape(samples[i], (samples[0].shape[0], 2))
        idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='red', alpha=.25)
        idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='green', alpha=.25)
        idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='blue', alpha=.25)
        idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='black', alpha=.25)
#        arr[i].set_xlim([-10, 10])
#        arr[i].set_ylim([-10, 10])
        arr[i].set_title(names[i])

# Observables
def compute_frequencies(f, I, prior):
    with tf.GradientTape() as g:
        g.watch(I)
        u = f.inverse(I)
        if prior == 'exponential':
            minus_log_prior = tf.reduce_sum( u , [1,2,3] )
        elif prior == 'normal':
            minus_log_prior = 0.5 * tf.reduce_sum( tf.square( u ) , [1,2,3] )
        minus_log_jac_det = f.log_jacobian_det( u )
        minus_log_rho = minus_log_prior + minus_log_jac_det
    return g.gradient(minus_log_rho, I)

def system_flow(q0, p0, T, f, ts, prior='exponential'):
    """Computes the system flow:
    phi_t(Q_0,P_0) = T( phi_t^Integrable( T.inverse(Q_0,P_0) ) )
    for t in ts. t is in the batch dimension. ts has shape (N,) """
    # Map to Q0, P0
    Q0, P0 = extract_q_p( T.inverse( join_q_p(q0,p0) ) )
    if prior == 'exponential' or prior == 'normal':
        # Compute frequencies in action-angle vars
        omega = compute_frequencies(f, P0, prior)
    elif prior == 'integrals_of_motion':
        # Here H = P_1, frequencies are trivial. Note the indexing
        # of the first component depends on that done in
        # BaseDistributionIntegralsOfMotion
        exp_minus_alpha1 = tf.reshape(tf.exp( - f.log_scale[0,0,0] ), (1,))
        sh = Q0.shape.as_list()
        omega = tf.concat([exp_minus_alpha1, tf.zeros(np.prod(sh)-1)],0)
        omega = tf.reshape(omega, sh)
    else:
        raise NotImplementedError
    # Motion in action angle variables
    sh = [ts.shape.as_list()[0], 1, 1, 1]
    Ps = tf.tile(P0, sh)
    Qs = Q0 + omega * tf.reshape(ts, sh)
    # Map back to original coordinates:
    qs, ps = extract_q_p( T( join_q_p(Qs, Ps) ) )
    return qs, ps, omega

def grad(f, vars):
    """Compute grad f(vars). vars is list"""
    # Branch code for eager vs lazy
    if tf.executing_eagerly():
        return tfe.gradients_function(f)(*vars)
    else:
        return tf.gradients(f(*vars), vars)

# Integrators
def euler(q0, p0, f, g, N, h):
    """Simple implementation of Euler integrator"""
    # Preallocate solutions
    n = q0.shape[0]
    qsol = np.zeros((N,n))
    psol = np.zeros((N,n))
    qsol[0,:] = q0
    psol[0,:] = p0
    for n in range(N-1):
        psol[n + 1,:] = psol[n,:] + h * f(qsol[n,:],psol[n,:])
        qsol[n + 1,:] = qsol[n,:] + h * g(qsol[n,:],psol[n,:])
    return qsol, psol

# TODO:
# 1. use the appropriate extract/join ops (DONE)
# 2. refactor to improve syntax, e.g. remove t1,x1 from rk4_step

# Define the solver step
def rk4_step(f, t, x, eps):
    k1 = f(t, x)
    k2 = f(t + eps/2, x + eps*k1/2)
    k3 = f(t + eps/2, x + eps*k2/2)
    k4 = f(t + eps, x + eps*k3)
    return eps/6 * (k1 + k2*2 + k3*2 + k4)

# Define the initial condition
def init_state(x,t,x0,t0):
    return tf.group(tf.assign(x, x0),
                    tf.assign(t, t0),
                    name='init_state')

# Define the update setp
def update_state(x,t,dx,eps):
    return tf.group(tf.assign_add(x, dx),
                    tf.assign_add(t, eps),
                    name='update_state')

# Hamiltonian vector field
def hamiltonian_vector_field(hamiltonian, t, x):
    # note, t unused
    q,p = extract_q_p(x)
    dq,dp = tf.gradients(hamiltonian(q,p),[q,p])
    return join_q_p(dp,-dq)
