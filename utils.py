"""
Utility functions
"""

import functools
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
tfe = tf.contrib.eager

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

# Tensor manipulation
def extract_q_p(x):
    """even/odd along the phase space axis"""
    #assert(x.shape[2] == 1)
    return x[:,::2,:], x[:,1::2,:]

def join_q_p(q, p):
    """join q, p of shape [N,n,1] to form x of shape [N,2*n,1] as follows
    [q0,p0,q1,p1,...]
    """
    #assert(q.shape == p.shape)
    #assert(q.shape[2] == 1)
    q_shape = q.shape
    return tf.reshape(tf.concat([q, p], 2), [-1, 2*q_shape[1], 1])

def split(x):
    """Split into two halves along the phase space dimension, returning z1,z2"""
    # Assume shape(x) = [batch, phase_space_dim, 1]
    x_shape = x.get_shape()
    phase_space_dim = int(x_shape[1])
    assert x_shape[2] == 1
    assert phase_space_dim % 2 == 0, "phase_space_dim: {}".format(phase_space_dim)
    return x[:, :phase_space_dim//2, :], x[:, phase_space_dim//2:, :]

def safe_concat(x, y, axis=1):
    """If x is empty, return y. If y is empty, return x."""
    if x.shape != 0 and y.shape != 0:
        return tf.concat([x,y], axis)
    elif x.shape != 0:
        return x
    elif y.shape != 0:
        return y
    else:
        return tf.constant([])

def lattice_shift(x):
    """x = (q_1,p_1,q_2,p_2, ..., q_n,p_n)
    -> (q_n,p_n,q_1,p_1, ..., q_{n-1},p_{n-1})"""
    q, p = extract_q_p(x)
    q_shifted = tf.manip.roll(q, shift=1, axis=1)
    p_shifted = tf.manip.roll(p, shift=1, axis=1)
    return join_q_p(q_shifted, p_shifted)

def is_symplectic(model, x):
    """Test if model is symplectic at x. Assumes x has shape (1,2*n,1)"""
    phase_space_dim = x.shape[1];
    with tf.GradientTape(persistent=True) as g:
        g.watch(x)
        y = tf.squeeze(model(x))
        # Need to wrap around a list to compute grad of elements. Why? tf bug?
        ys = [y[i] for i in range(phase_space_dim)]
    # Do the rest in numpy
    jacobian = np.zeros((phase_space_dim, phase_space_dim))
    for i in range(phase_space_dim):
        # set the i-th row
        jacobian[i,:] = np.reshape(g.gradient(ys[i], x).numpy(), (phase_space_dim,))
    del g

    iSigma2 = np.array([[0,1],[-1,0]])
    omega = np.kron(np.eye(phase_space_dim//2), iSigma2)
    omega_tilde = np.dot(np.dot(jacobian, omega), np.transpose(jacobian))
    return np.allclose(omega_tilde, omega, rtol=1e-05, atol=1e-08)

# Training utils
def compute_loss(model, hamiltonian, z):
    return tf.reduce_mean(hamiltonian(model(z)))

def compute_gradients(model, hamiltonian, z):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, hamiltonian, z)
    return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables, global_step=None):
    optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

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
    assert(tf.reduce_all(tf.equal(a,b)))

def assert_allclose(x, y, rtol=1e-5, atol=1e-8):
    assert(tf.reduce_all(tf.abs(x - y) <= tf.abs(y) * rtol + atol))

# Visualization
def visualize_chain_bijector_1d(model, x):
    """Assumes eager mode"""
    samples = [x]
    names = ["base_dist"]
    for bijector in model.bijectors:
        x = bijector(x)
        samples.append(x)
        names.append(bijector.name)
    f, arr = plt.subplots(1, len(samples), figsize=(4 * (len(samples)), 4))
    X0 = tf.reshape(samples[0].numpy(), shape=(samples[0].shape[0], 2))
    for i in range(len(samples)):
        X1 = tf.reshape(samples[i].numpy(), shape=(samples[0].shape[0], 2)).numpy()
        idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='red')
        idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='green')
        idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='blue')
        idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='black')
        arr[i].set_xlim([-10, 10])
        arr[i].set_ylim([-10, 10])
        arr[i].set_title(names[i])

# Observables
def system_flow(Q_0, P_0, model, ts):
    """Computes the system flow:
    phi_t(Q_0,P_0) = model( phi_t^osc( model.inverse(Q_0,P_0) ) )
    for t in ts. t is in the batch dimension.
    Here Q_0 and P_0 are 1-d tensors of length n."""
    # Reshape data
    n = Q_0.shape[0]
    Q_0 = tf.reshape(Q_0, [1,n,1])
    P_0 = tf.reshape(P_0, [1,n,1])
    ts = tf.reshape(ts, shape=(ts.size,1,1))

    # Invert the initial conditions
    q_0, p_0 = extract_q_p(model.inverse(join_q_p(Q_0, P_0)))
    alpha = tf.atan(q_0/p_0)
    sqrt2E = q_0/tf.sin(alpha)

    # Evolve the initial conditions
    qt = sqrt2E * tf.sin(ts + alpha)
    pt = sqrt2E * tf.cos(ts + alpha)
    return model(join_q_p(qt, pt))

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
