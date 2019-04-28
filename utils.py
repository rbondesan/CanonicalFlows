"""
Utility functions
"""

import functools
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
tfe = tf.contrib.eager
from tensorflow.python.ops.parallel_for import gradients as tf_gradients_ops

DTYPE = tf.float32
NP_DTYPE=np.float32

# Tensor manipulation
def extract_q_p(x):
    """split along the channel axis: q = x[...,0], p = x[...,1]"""
    #tf.assert_equal(tf.shape(x)[-1], tf.constant(2))
    # Use :: to keep number of dimensions
    return x[...,::2], x[...,1::2]

def join_q_p(q, p):
    """join q,p along channel axis"""
    #tf.assert_equal(tf.shape(q), tf.shape(p))
    #tf.assert_equal(tf.shape(q)[-1], tf.constant(1))
    return tf.concat([q, p], -1)

def get_phase_space_dim(sh):
    """sh = (N,n1,n2,...,nd,2) -> n1 * n2 * ... * nd * 2"""
    return tf.reduce_prod(sh[1:])

def int_shape(x):
    if str(x.get_shape()[0]) != '?':
        return list(map(int, x.get_shape()))
    return [-1]+list(map(int, x.get_shape()[1:]))

def normsq_nobatch(x):
    return tf.reduce_sum( tf.square(x), [1,2,3] )

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


# TODO: update
# Symmetry utils
# def lattice_shift(x):
#     """x = (q_1,p_1,q_2,p_2, ..., q_n,p_n)
#     -> (q_n,p_n,q_1,p_1, ..., q_{n-1},p_{n-1})"""
#     q, p = extract_q_p(x)
#     q_shifted = tf.manip.roll(q, shift=1, axis=1)
#     p_shifted = tf.manip.roll(p, shift=1, axis=1)
#     return join_q_p(q_shifted, p_shifted)

def make_train_op(settings, loss, step):
    with tf.name_scope("train"):
        starter_learning_rate = settings['starter_learning_rate']
        if settings['decay_lr'] == "exp":
            learning_rate = tf.train.exponential_decay(starter_learning_rate, step, settings['decay_steps'],
                                                       settings['decay_rate'], staircase=False)
        elif settings['decay_lr'] == "piecewise":
            boundaries = settings['boundaries']
            values = settings['values']
            learning_rate = tf.train.piecewise_constant(step, boundaries, values)
        else:
            learning_rate = tf.constant(starter_learning_rate)
        learning_rate = tf.maximum(learning_rate, settings['min_learning_rate']) # clip
        if settings['visualize']:
            tf.summary.scalar("lr", learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss=loss)
        if 'grad_clip_norm' in settings:
            grads_and_vars = [(tf.clip_by_norm(gv[0], settings['grad_clip_norm']), 
                               gv[1]) for gv in grads_and_vars]
        
        if settings['visualize']:
            for gradient, variable in grads_and_vars:
                tf.summary.scalar("gradients/" + variable.name.replace(':','_'), tf.norm(gradient))
                tf.summary.scalar("variable/" + variable.name.replace(':','_'), tf.norm(variable))
    
    return optimizer.apply_gradients(grads_and_vars, global_step=step)

def compute_jacobian_eager(model, x):
    """Test if model is simplectic at x.
    Assume x.shape = (1,n1,n2,..,nd,2)"""
    x_shape = x.shape
    phase_space_dim = np.prod(x_shape[1:])
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
    return jacobian

def compute_np_jacobian_lazy(y, x, sess):
    """Return is a np matrix"""
    J = tf_gradients_ops.jacobian(y,x,use_pfor=False) # Need pfor false, not sure why...
    phase_space_dim = tf.reduce_prod(tf.shape(x)[1:])
    J_np = sess.run( tf.reshape(J, (phase_space_dim, phase_space_dim)) )
    return J_np

def is_symplectic(model, x, sess=None, rtol=1e-05, atol=1e-08):
    """Test if model is simplectic at x in numpy.
    Assume x.shape = (1,n1,n2,..,nd,2)"""
    if tf.executing_eagerly():
        #import pdb; pdb.set_trace()
        J = compute_jacobian_eager(model, x)
    else:
        J = compute_np_jacobian_lazy(model(x), x, sess)
    phase_space_dim = J.shape[0]
    iSigma2 = np.array([[0,1],[-1,0]])
    omega = np.kron(np.eye(phase_space_dim//2), iSigma2)
    omega_tilde = np.dot(np.dot(J, omega), np.transpose(J))
    return np.allclose(omega_tilde, omega, rtol=rtol, atol=atol)

def generate_and_save_images(model, epoch, test_input, sess, save=False):
    predictions = model(test_input)
#    fig = plt.figure(figsize=(4,4))
    fig = plt.figure()
    visualize_chain_bijector_1d(model, test_input, sess=sess)
    if save:
        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

def checkpoint_save(settings, optimizer, model, optimizer_step):
    checkpoint_dir = settings['log_dir']
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

def visualize_chain_bijector(model, x, sess=None, inverse=False):
    assert not tf.executing_eagerly()

    # Compute data
    samples = [sess.run(x)]
    names = ["base_dist"]
    if inverse:
        for bijector in reversed(model.bijectors):
            x = bijector.inverse(x)
            samples.append(sess.run(x))
            names.append(bijector.name)
    else:
        for bijector in model.bijectors:
            x = bijector(x)
            samples.append(sess.run(x))
            names.append(bijector.name)

    # Subplot with nrows = d * num_particles, cols = number of bijectors
    d = samples[0].shape[1]
    num_particles = samples[0].shape[2]
    nrows = d * num_particles
    ncols = len(samples)
    f, arr = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1:
        arr = np.expand_dims(arr,0)

    for a in range(d):
        for b in range(num_particles):
            row = a * num_particles + b
            # X0 is the first column
            X0 = np.reshape(samples[0][:,a,b], (samples[0].shape[0], 2))
            # Go throught the columns
            for i in range(len(samples)):
                X1 = np.reshape(samples[i][:,a,b], (samples[i].shape[0], 2))

                idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
                arr[row,i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='red',
                                   alpha=.25)
                idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
                arr[row,i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='green',
                                   alpha=.25)
                idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
                arr[row,i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='blue',
                                   alpha=.25)
                idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
                arr[row,i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='black',
                                   alpha=.25)
        #        arr[row,i].set_xlim([-10, 10])
        #        arr[rowi].set_ylim([-10, 10])
                arr[row,i].set_title(names[i]+"_"+str(a)+"_"+str(b))

# Observables
def compute_gradK_penalty(K, z):
    # Note: this works only in graph mode.
    dPsi, dF = extract_q_p( tf.gradients(K, z)[0] )
    dF = tf.layers.flatten(dF)
    penaltyF = tf.square( dF[:,0] - 1 ) + tf.reduce_sum(tf.square( dF[:,1:] ), 1)
    penaltyPsi = tf.reduce_sum(tf.square(dPsi), [1,2,3])
    return tf.reduce_mean(penaltyF + penaltyPsi)

# def indicator_fcn(low, high, x):
#     # return: 0 if x in [low, high]; 1 otherwise
#     return tf.cast(tf.reduce_all(tf.logical_and(low < x, x < high)), DTYPE)

def confining_potential(x, low, high):
    # Penalizes x outside the range [low,high]:
    # -V(x-low) if x < low
    # 0         if low < x < high
    # V(x-high) if x > high
    # Use linear potential.
    V = lambda x : x
    return tf.where(tf.greater(x, high), V(x-high),                # x > high
                    tf.where(tf.greater(x, low), tf.zeros_like(x), # low < x < high
                             -V(x-low)))                           # x < low

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

def hamiltons_equations(H,settings):
    """Obtain Hamilton's equations by autodiff, assuming phase space point in
    format [1,d,n,2]"""
    def flow(phase_space_point, t):
        phase_space_point = tf.reshape(phase_space_point, [1,settings['d'],settings['num_particles'],2])
        H_grads = tf.gradients(H(phase_space_point), phase_space_point)[0]
        dHdq, dHdp = extract_q_p(H_grads)
        flow_vec = join_q_p(dHdp, -dHdq)
        return tf.reshape(flow_vec, (settings['d']*settings['num_particles']*2,))
    return flow

def hamiltonian_traj(H, init_state, settings, time=100, steps=200, rtol=1e-04, atol=1e-6):
    """Integrate Hamilton's equations using TF, given initial phase space point"""
    t = tf.linspace(0.0, time, num=steps)
    tensor_state = tf.contrib.integrate.odeint(hamiltons_equations(H,settings), init_state, t, rtol, atol)
    return tensor_state

# Traj2Circle utils
def plot_traj(settings, qtraj, ptraj, qhat_traj=None, phat_traj=None, equal_aspect=True):    
    cols=['r','g','b']        
    num_init_cond = qtraj.shape[1]
    fig = plt.figure(figsize=(4*settings['d'],4*num_init_cond))    
    for b in range(num_init_cond):
        for n in range(settings['d']):
            plt.subplot(num_init_cond, settings['d'], n + b*settings['d'] + 1) #nrows,ncols,index
            plt.plot(qtraj[:,b,n,0,0], ptraj[:,b,n,0,0], '+')
            if qhat_traj is not None and phat_traj is not None:
                plt.plot(qhat_traj[:,b,n,0,0], phat_traj[:,b,n,0,0], '*')
            if equal_aspect:
                plt.gca().set_aspect('equal', adjustable='box')

def pull_back_traj(settings, T, x):
    """Returns tranformed trajectory x whose shape is (T,B,d,n,2)."""
    batch = x.shape[1]    
    z = tf.reshape(x, [settings['minibatch_size']*batch,settings['d'],settings['num_particles'],2]) 
    z = T.inverse(z)
    z = tf.reshape(z, [settings['minibatch_size'],batch,settings['d'],settings['num_particles'],2])
    return z


# TODO: Remove?
# # Define the solver step
# def rk4_step(f, t, x, eps):
#     k1 = f(t, x)
#     k2 = f(t + eps/2, x + eps*k1/2)
#     k3 = f(t + eps/2, x + eps*k2/2)
#     k4 = f(t + eps, x + eps*k3)
#     return eps/6 * (k1 + k2*2 + k3*2 + k4)
#
# # Define the initial condition
# def init_state(x,t,x0,t0):
#     return tf.group(tf.assign(x, x0),
#                     tf.assign(t, t0),
#                     name='init_state')
#
# # Define the update setp
# def update_state(x,t,dx,eps):
#     return tf.group(tf.assign_add(x, dx),
#                     tf.assign_add(t, eps),
#                     name='update_state')
#
# # Hamiltonian vector field
# def hamiltonian_vector_field(hamiltonian, t, x):
#     # note, t unused
#     q,p = extract_q_p(x)
#     dq,dp = tf.gradients(hamiltonian(q,p),[q,p])
#     return join_q_p(dp,-dq)
