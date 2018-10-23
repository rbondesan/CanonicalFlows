"""
Hamiltonians. x is a tensor of shape (batch, phase_space_dim, 1).
The Hamiltonian returns a tensor of shape (batch).
"""

import numpy as np
import tensorflow as tf

from utils import extract_q_p

# 1 particle in 1d.
def toy_hamiltonian(x):
    """1/2 * (q-1/4 p^2)^2 + 1/32 p^2"""
    assert(x.shape[2] == 1)
    q,p = extract_q_p(x)
    pSqr = tf.square(p)
    return 1/2 * tf.square(q - 1/4 * pSqr) + 1/32 * pSqr

def pendulum_hamiltonian(x):
    """1/2 * p^2 + cos(q)"""
    assert(x.shape[2] == 1)
    q,p = extract_q_p(x)
    return 1/2 * tf.square(p) + tf.cos(q)

# Integrable many particle
def parameterized_neumann(ks):
    def neumann_hamiltonian(x):
        """1/4 \sum_{i,j}^N J_{ij}^2 + 1/2 \sum_{i=1}^N k_i q_i^2"""
        assert (x.shape[2] == 1)
        q, p = extract_q_p(x)
        J = tf.einsum('bi,bj->bij', q, p)
        return tf.einsum('bij,bji->b', J, J) / 4 + tf.einsum('bi,bi->b', ks, tf.square(q)) / 2
    return neumann_hamiltonian

# Chains
def oscillator_hamiltonian(x):
    """Harmonic oscillator: 1/2 \sum_{i=1}^N p_i^2 + q_i^2
    x.shape = (batch, phase_space, 1)
    """
    assert(x.shape[2] == 1)
    q, p = extract_q_p(x)
    return 0.5 * tf.reduce_sum(tf.square(p) + tf.square(q),  axis=1)

def oscillator_diff_hamiltonian(x):
    """Harmonic oscillator: 1/2 \sum_{i=1}^N p_i^2 + q_i^2
    x.shape = (batch, phase_space, 1)
    """
    assert(x.shape[2] == 1)
    q, p = extract_q_p(x)
    qdiff = q - tf.manip.roll(q, shift=-1, axis=1) # q - (q_2, q_3, ..., q_{N}, q_1)
    return 0.5 * tf.reduce_sum(tf.square(p) + tf.square(qdiff),  axis=1)

def diff_square_hamiltonian(x):
    """1/2 \sum_{i=1}^N  [p_i^2 + (q_{i} - q_{i+1})^2] (with q_{N+1} = q_1)
    x.shape = (batch, phase_space, 1)
    """
    assert(x.shape[2] == 1)
    q, p = extract_q_p(x)
    qdiff = q - tf.manip.roll(q, shift=-1, axis=1) # q - (q_2, q_3, ..., q_{N}, q_1)
    return 0.5 * tf.reduce_sum(tf.square(p) + tf.square(qdiff),  axis=1)

def fpu_alpha_hamiltonian(x):
    """\sum_{i=1}^N 1/2 [p_i^2 + (q_{i} - q_{i+1})^2] + 1/3 (q_{i} - q_{i+1})^3
    (with q_{N+1} = q_1). x.shape = (batch, phase_space, 1)
    """
    assert(x.shape[2] == 1)
    q, p = extract_q_p(x)
    qdiff = q - tf.manip.roll(q, shift=-1, axis=1) # q - (q_2, q_3, ..., q_{N}, q_1)
    return tf.reduce_sum(0.5 * tf.square(p) + 0.5 * tf.square(qdiff) + 1/3. * tf.pow(qdiff, 3),  axis=1)

def fpu_beta_hamiltonian(x):
    """\sum_{i=1}^N 1/2 [p_i^2 + (q_{i} - q_{i+1})^2] + 1/4 (q_{i} - q_{i+1})^4
    (with q_{N+1} = q_1) x.shape = (batch, phase_space, 1)
    """
    assert(x.shape[2] == 1)
    q, p = extract_q_p(x)
    qdiff = q - tf.manip.roll(q, shift=-1, axis=1) # q - (q_2, q_3, ..., q_{N}, q_1)
    return tf.reduce_sum(0.5 * tf.square(p) + 0.5 * tf.square(qdiff) + 1/4. * tf.pow(qdiff, 4),  axis=1)

def toda_hamiltonian(x):
    """Toda lattice: 1/2 \sum_{i=1}^N  [p_i^2 + exp^{q_{i} - q_{i+1}}] (with q_{N+1} = q_1)
    x.shape = (batch, phase_space, 1)
    """
    assert(x.shape[2] == 1)
    q, p = extract_q_p(x)
    # TODO: Compute exp(qdiff) avoiding overfloating (which occurs for exp(hundred) or so)
    qdiff = q - tf.manip.roll(q, shift=-1, axis=1) # q - (q_2, q_3, ..., q_{N}, q_1)
    return 0.5 * tf.reduce_sum(tf.square(p) + tf.exp(qdiff),  axis=1)
