"""
Tests for functions in hamiltonians
"""

import tensorflow as tf
from hamiltonians import *
from utils import assert_equal

tf.enable_eager_execution()

DTYPE = tf.float32

def test_toy_hamiltonian():
    x = tf.constant([2.,3.], shape=(1,2,1))
    h = toy_hamiltonian(x)
    expected_h = tf.constant(1/2 * (2 - 1/4 * 3**2)**2 + 1/2 * 3**2 / 16)
    assert_equal(h, expected_h)
    print('test_toy_hamiltonian passed')
test_toy_hamiltonian()

# TODO: Test pendulum_hamiltonian

def test_henon_heiles_hamiltonian():
    x = tf.constant([1.,2.,3.,4.], shape=(1,4,1)) # q1 = 1, p1 = 2, q2 = 3, p2 = 4
    h = henon_heiles_hamiltonian(x)
    expected_h = tf.constant(1/2 * (2**2 + 4**2 + 1 + 3**2) + 1*3 - 1/3 * 3**3)
    assert_equal(h, expected_h)

    a, b, c, d = .2, .3, .4, .5
    h = henon_heiles_hamiltonian(x, a=a, b=b, c=c, d=d)
    expected_h = tf.constant(1/2 * (2**2 + 4**2 + a*1 + b*3**2) + d*1*3 - 1/3 *c* 3**3)
    assert_equal(h, expected_h)

    print('test_henon_heiles_hamiltonian passed')
test_henon_heiles_hamiltonian()

def test_oscillator_hamiltonian():
    x = tf.reshape(tf.range(1,5,dtype=DTYPE),[1,4,1]) # q = [1,2], p=[3,4]
    h = oscillator_hamiltonian(x)
    expected_h = tf.constant(1/2 * (9 + 16 + 1 + 4), dtype=DTYPE)
    assert_equal(h, expected_h)
    print('test_oscillator_hamiltonian passed')
test_oscillator_hamiltonian()

def test_diff_square_hamiltonian():
    x = tf.reshape(tf.range(1,7,dtype=DTYPE),[1,6,1]) # q = [1,3,5], p=[2,4,6]
    h = diff_square_hamiltonian(x)
    # expected = 1/2 * (4 + 16 + 36 + (1.-3.)**2 + (3.-5.)**2 + (5.-1.)**2)
    expected_h = tf.constant(1/2 * (4 + 16 + 36 + 4 + 4 + 16), dtype=DTYPE)
    assert_equal(h, expected_h)
    print('test_diff_square_hamiltonian passed')
test_diff_square_hamiltonian()

def test_fpu_alpha_hamiltonian():
    x = tf.reshape(tf.range(1,7,dtype=DTYPE),[1,6,1]) # q = [1,3,5], p=[2,4,6]
    h = fpu_alpha_hamiltonian(x)
    expected_h = tf.constant(1/2 * (4 + 16 + 36 + 4 + 4 + 16) + 1/3 * (- 8 - 8 + 64), dtype=DTYPE)
    assert_equal(h, expected_h)
    print('test_fpu_alpha_hamiltonian passed')
test_fpu_alpha_hamiltonian()

def test_fpu_beta_hamiltonian():
    x = tf.reshape(tf.range(1,7,dtype=DTYPE),[1,6,1]) # q = [1,3,5], p=[2,4,6]
    h = fpu_beta_hamiltonian(x)
    expected_h = tf.constant(1/2 * (4 + 16 + 36 + 4 + 4 + 16) + 1/4 * (16 + 16 + 4**4), dtype=DTYPE)
    assert_equal(h, expected_h)
    print('test_fpu_beta_hamiltonian passed')
test_fpu_beta_hamiltonian()

def test_toda_hamiltonian():
    x = tf.reshape(tf.range(1,7,dtype=DTYPE),[1,6,1]) # q = [1,3,5], p=[2,4,6]
    h = toda_hamiltonian(x)
    expected_h = tf.constant(
        1/2 * (4 + 16 + 36 + np.exp(1.-3.) + np.exp(3.-5.) + np.exp(5.-1.)),
        dtype=DTYPE)
    assert_equal(h, expected_h)
    print('test_toda_hamiltonian passed')
test_toda_hamiltonian()
