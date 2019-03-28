"""
Hamiltonians. x is a tensor of shape (batch, phase_space_dim, 1).
The Hamiltonian returns a tensor of shape (batch).
"""

import numpy as np
import tensorflow as tf

from utils import extract_q_p

# # 1 particle in 1d.
def toy_hamiltonian(x):
    """1/2 * (q-1/4 p^2)^2 + 1/32 p^2
    Assume x.shape = (N,1,n,2) with n=1,2,..."""
    q,p = extract_q_p(x)
    pSqr = tf.square(p)
    return 1/2 * tf.square(q - 1/4 * pSqr) + 1/32 * pSqr

# def pendulum_hamiltonian(x):
#     """1/2 * p^2 + cos(q)"""
#     assert(x.shape[2] == 1)
#     q,p = extract_q_p(x)
#     return 1/2 * tf.square(p) + tf.cos(q)

def harmonic_oscillator(x):
    """Harmonic oscillator: 1/2 sum( p^2 + q^2 ) Assume x.shape =
    (N,d,n,2), n uncoupled harmonic oscillators in d-dimensions

    """
    q, p = extract_q_p(x)
    return 0.5 * tf.reduce_sum(tf.square(p) + tf.square(q),  axis=[1,2,3])

def kepler(x, k=1.0):
    """H = 1/2 sum_{i=1}^d p_i^2 + k/r, r = sqrt(sum_{i=1}^d q_i^2).
    Assume x.shape = (N,d,1,2) with d=2,3.
    V(r)=k/r, k>0 repulsive, k<0 attractive."""
    q,p = extract_q_p(x)
    # The derivative of r wrt q is 1/sqrt(sum(q^2)), which is singular in 0.
    # Cutoff r so that it is > eps.
    eps = 1e-5
    r = tf.sqrt(tf.reduce_sum(tf.square(q), axis=1) + eps)
    return tf.squeeze(0.5 * tf.reduce_sum(tf.square(p), axis=1) + k / r)

def free(x):
    """H = 1/2 sum_{i=1}^d p_i^2.
    Assume x.shape = (N,d,1,2) with d=1,2,3,..."""
    _, p = extract_q_p(x)
    return tf.squeeze(0.5 * tf.reduce_sum(tf.square(p), axis=1))

# Integrable many particle
def parameterized_neumann(ks):
    def neumann_hamiltonian(x):
        """1/4 \sum_{i,j}^N J_{ij}^2 + 1/2 \sum_{i=1}^N k_i q_i^2"""
        # ks is of shape (d,)
        assert x.shape[2] == 1
        q, p = extract_q_p(x)
        q = q[:,:,0,0]
        p = p[:,:,0,0]
        # q,p of shape (N,d)
        J = tf.einsum('ai,aj->aij', q, p)
        J -= tf.transpose(J, perm=[0,2,1])
        # J is of shape (N,d,d)
        return tf.squeeze(
                0.25 * tf.reduce_sum(tf.square(J), axis=[1,2]) + \
                0.5 * tf.reduce_sum( tf.multiply(ks, tf.square(q)), axis=1 ))
    return neumann_hamiltonian

def fpu_hamiltonian(x, alpha=1, beta=0):
    """\sum_{i=1}^N 1/2 [p_i^2 + (q_{i} - q_{i+1})^2] + \alpha/3 (q_{i} - q_{i+1})^3 + \beta/4 (q_{i} - q_{i+1})^4
    (with q_{N+1} = q_1). x.shape = (batch, 1, n, 2)
    """
    assert(x.shape[1] == 1)
    q, p = extract_q_p(x)
    qdiff = q - tf.manip.roll(q, shift=-1, axis=2) # q - (q_2, q_3, ..., q_{N}, q_1)
    h = tf.reduce_sum(0.5 * tf.square(p) + 0.5 * tf.square(qdiff)
                      + alpha / 3. * tf.pow(qdiff, 3)
                      + beta / 4. * tf.pow(qdiff, 4),  axis=2)
    return h

def open_toda(x):
    """1/2 \sum_{i=1}^N  p_i^2 + \sum_{i=1}^{n-1} exp(q_i - q_{i+1}).
    x.shape = (N,1,n,2)"""
    q, p = extract_q_p(x)
    # q2, q3, ... , qN, q1
    qshift = tf.manip.roll(q, shift=-1, axis=2)
    # q1-q2, q2-q3, ... , q{N-1}-qN -> omit qN-q1, so qdiff shape (N,1,n-1,1)
    qdiff = q[:,:,:-1,:] - qshift[:,:,:-1,:]
    V = tf.reduce_sum(tf.exp(qdiff), axis=2)
    K = 0.5 * tf.reduce_sum(tf.square(p), axis=2)
    return K + V

def closed_toda(x):
    """1/2 \sum_{i=1}^n  p_i^2 + \sum_{i=1}^{n} exp(q_i - q_{i+1}).
    x.shape = (N,1,n,2)"""
    q, p = extract_q_p(x)
    # q2, q3, ... , qN, q1
    qshift = tf.manip.roll(q, shift=-1, axis=2)
    # q1-q2, q2-q3, ... , q{N-1}-qN,qN-q1
    qdiff = q - qshift
    return tf.reduce_sum(0.5 * tf.square(p)+tf.exp(qdiff), axis=2)

def closed_toda_3(x):
    """p_x^2 + p_y^2 + e^{-2 y}+ e^{y-\sqrt{3}x}+ e^{y+\sqrt{3}x}\,.
    x.shape = (N,1,2,2). Normal mode expression if center mass fixed."""
    q, p = extract_q_p(x)
    x = q[:,0,0,0]
    y = q[:,0,1,0]
    V = tf.exp(-2.*y) + tf.exp(y - tf.sqrt(3.)*x) + tf.exp(y + tf.sqrt(3.)*x)
    return tf.reduce_sum(tf.square(p), axis=2) + V

def calogero_moser(x, type, omegasq=1., gsq=1., mu=1.):
    """
    Notation: https://www1.maths.leeds.ac.uk/~siru/papers/p38.pdf (2.38)
    and http://www.scholarpedia.org/article/Calogero-Moser_system
    
    H = \frac{1}{2}\sum_{i=1}^n (p_i^2 + \omega^2 q_i^2) + g^2 \sum_{1\le j < k \le n} V(q_j - q_k)
    
    V(x) = 
    'rational':      1/x^2
    'hyperbolic':    \mu^2/4\sinh(\mu x/2)
    'trigonometric': \mu^2/4\sin(\mu x/2)
    """
    assert(x.shape[1] == 1)

    if type == 'rational':
        V = lambda x : 1/x**2
    elif type == 'hyperbolic':
        V = lambda x : mu**2/4./tf.sinh(mu/2. * x)
    elif type == 'trigonometric':
        V = lambda x : mu**2/4./tf.sin(mu/2. * x)
    else:
        raise NotImplementedError
    
    q, p = extract_q_p(x)
    h_free = 0.5 * tf.reduce_sum(tf.square(p) + omegasq * tf.square(q), axis=[1,2,3]) # (batch,)
        
    # Compute matrix of deltaq q[i]-q[j] and extract upper triangular part (triu)
    q = tf.squeeze(q, 1) # (N,n,1)
    deltaq = tf.transpose(q, [0, 2, 1]) - q # (N,n,n)
    n = tf.shape(deltaq)[1]
    ones = tf.ones([n,n])
    triu_mask = tf.cast(tf.matrix_band_part(ones, 0, -1) - \
                        tf.matrix_band_part(ones, 0, 0), dtype=tf.bool)    
    triu = tf.boolean_mask(deltaq, triu_mask, axis=1)    
    eps = 1e-5 # regulizer for inverse
    h_int = gsq * tf.reduce_sum(V(triu + eps), axis=1) # (batch,)
    
    return h_free + h_int

# sum_{i<j} V(x(i) - x(j)) : 
    # e.g.: 
    # x = np.arange(10)
    # x = np.expand_dims(x, 1)
    # diffs = np.transpose(x) - x
    # idx = np.triu_indices(np.shape(diffs)[1],k=1) 
    # np.sum(V(diffs[idx]))