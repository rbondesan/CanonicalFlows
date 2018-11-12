"""
Tests for functions in hamiltonians
"""

import tensorflow as tf
from hamiltonians import *
from utils import assert_equal, run_eagerly


DTYPE = tf.float32

@run_eagerly
def test_toy_hamiltonian():
    x = tf.constant([2.,3.], shape=(1,2,1))
    h = toy_hamiltonian(x)
    expected_h = tf.constant(1/2 * (2 - 1/4 * 3**2)**2 + 1/2 * 3**2 / 16)
    print(expected_h.shape, h.shape)
    assert_equal(h, expected_h)

# TODO: Test pendulum_hamiltonian

@run_eagerly
def test_parameterized_neumann():
    N = 3
    ks = tf.random_uniform([N])
    neumann_hamiltonian = parameterized_neumann(ks)
    x = tf.random_uniform([1, 2*N, 1])
    h = neumann_hamiltonian(x)
    q = x[0,::2,0]
    p = x[0, 1::2, 0]
    J = tf.einsum('i,j->ij', q, p)
    J -= tf.transpose(J)
    expected_h = tf.reduce_sum(tf.square(J)) / 4 + tf.reduce_sum(ks * tf.square(q)) / 2
    assert_equal(h, expected_h)

@run_eagerly
def test_parameterized_oscillator():
    N = 3
    ks = tf.random_uniform([N])
    oscillator_hamiltonian = parameterized_oscillator(ks)
    x = tf.random_uniform([1, 2*N, 1])
    h = oscillator_hamiltonian(x)
    q = x[0,::2,0]
    p = x[0, 1::2, 0]
    expected_h = tf.reduce_sum(tf.square(p) + ks * tf.square(q)) / 2
    assert_equal(h, expected_h)

@run_eagerly
def test_henon_heiles_hamiltonian():
    x = tf.constant([1.,2.,3.,4.], shape=(1,4,1)) # q1 = 1, p1 = 2, q2 = 3, p2 = 4
    h = henon_heiles_hamiltonian(x)
    expected_h = tf.constant(1/2 * (2**2 + 4**2 + 1 + 3**2) + 1*3 - 1/3 * 3**3)
    assert_equal(h, expected_h)

    a, b, c, d = .2, .3, .4, .5
    h = henon_heiles_hamiltonian(x, a=a, b=b, c=c, d=d)
    expected_h = tf.constant(1/2 * (2**2 + 4**2 + a*1 + b*3**2) + d*1*3 - 1/3 *c* 3**3)
    assert_equal(h, expected_h)

@run_eagerly
def test_diff_square_hamiltonian():
    x = tf.reshape(tf.range(1,7,dtype=DTYPE),[1,6,1]) # q = [1,3,5], p=[2,4,6]
    h = diff_square_hamiltonian(x)
    # expected = 1/2 * (4 + 16 + 36 + (1.-3.)**2 + (3.-5.)**2 + (5.-1.)**2)
    expected_h = tf.constant(1/2 * (4 + 16 + 36 + 4 + 4 + 16), dtype=DTYPE)
    assert_equal(h, expected_h)

@run_eagerly
def test_fpu_alpha_hamiltonian():
    x = tf.reshape(tf.range(1,7,dtype=DTYPE),[1,6,1]) # q = [1,3,5], p=[2,4,6]
    h = fpu_alpha_hamiltonian(x)
    expected_h = tf.constant(1/2 * (4 + 16 + 36 + 4 + 4 + 16) + 1/3 * (- 8 - 8 + 64), dtype=DTYPE)
    assert_equal(h, expected_h)

@run_eagerly
def test_fpu_beta_hamiltonian():
    x = tf.reshape(tf.range(1,7,dtype=DTYPE),[1,6,1]) # q = [1,3,5], p=[2,4,6]
    h = fpu_beta_hamiltonian(x)
    expected_h = tf.constant(1/2 * (4 + 16 + 36 + 4 + 4 + 16) + 1/4 * (16 + 16 + 4**4), dtype=DTYPE)
    assert_equal(h, expected_h)

@run_eagerly
def test_toda_hamiltonian():
    x = tf.reshape(tf.range(1,7,dtype=DTYPE),[1,6,1]) # q = [1,3,5], p=[2,4,6]
    h = toda_hamiltonian(x)
    expected_h = tf.constant(
        1/2 * (4 + 16 + 36 + np.exp(1.-3.) + np.exp(3.-5.) + np.exp(5.-1.)),
        dtype=DTYPE)
    assert_equal(h, expected_h)
