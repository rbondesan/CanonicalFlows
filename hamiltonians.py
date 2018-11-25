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

def single_harmonic_oscillator(x):
    """Harmonic oscillator: 1/2 ( p^2 + q^2 )
    Assume x.shape = (N,1,1,2), d=n=1."""
    q, p = extract_q_p(x)
    return 0.5 * tf.reduce_sum(tf.square(p) + tf.square(q),  axis=[1,2,3])

# 1 particle in d-dim.
def coulomb(r):
    """r.shape = (N,1,1) """
    eps = 1e-5
    return 1.0 / (r + eps)

def kepler(x, V=coulomb):
    """H = 1/2 sum_{i=1}^d p_i^2 + V(r), r = sqrt(sum_{i=1}^d q_i^2).
    Assume x.shape = (N,d,1,2) with d=2,3."""
    q,p = extract_q_p(x)
    r = tf.sqrt(tf.reduce_sum(tf.square(q), axis=1))
    return tf.squeeze(0.5 * tf.reduce_sum(tf.square(p), axis=1) + V(r))

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

# TODO: update
# # Chains
# def toy_hamiltonian_chain(x):
#     """1/2 * (q-1/4 p^2)^2 + 1/32 p^2. Decoupled"""
#     assert(x.shape[2] == 1)
#     q,p = extract_q_p(x)
#     pSqr = tf.square(p)
#     return 1/2 * tf.reduce_sum( tf.square(q - 1/4 * pSqr) + 1/32 * pSqr , axis=1 )
#
#
# def diff_square_hamiltonian(x, eps=0):
#     """1/2 \sum_{i=1}^N  [p_i^2 + eps * q_i^2 + (q_{i} - q_{i+1})^2] (with q_{N+1} = q_1)
#     x.shape = (batch, phase_space, 1)
#     Here eps is a regulator to make the density e^-H normalizable, avoiding zero mode.
#     """
#     assert(x.shape[2] == 1)
#     q, p = extract_q_p(x)
#     qdiff = q - tf.manip.roll(q, shift=-1, axis=1) # q - (q_2, q_3, ..., q_{N}, q_1)
#     return 0.5 * tf.reduce_sum(tf.square(p) + + 0.5 * eps * tf.square(q) + tf.square(qdiff),  axis=1)
#
# def toda_hamiltonian(x):
#     """Toda lattice: 1/2 \sum_{i=1}^N  [p_i^2 + exp^{q_{i} - q_{i+1}}] (with q_{N+1} = q_1)
#     x.shape = (batch, phase_space, 1)
#     """
#     assert(x.shape[2] == 1)
#     q, p = extract_q_p(x)
#     # TODO: Compute exp(qdiff) avoiding overfloating (which occurs for exp(hundred) or so)
#     # TODO: Add regulator eps
#     qdiff = q - tf.manip.roll(q, shift=-1, axis=1) # q - (q_2, q_3, ..., q_{N}, q_1)
#     return 0.5 * tf.reduce_sum(tf.square(p) + tf.exp(qdiff),  axis=1)
