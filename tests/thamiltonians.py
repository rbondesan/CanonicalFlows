"""
Tests for functions in hamiltonians
"""

import tensorflow as tf
import sys
sys.path.append("../")
from hamiltonians import *
from utils import assert_equal, assert_allclose

tf.enable_eager_execution()

DTYPE = tf.float32

def test_kepler():
    # q=1,3,5; p=2;4;6
    # q=7,9,11; p=8;10;12
    x = tf.reshape(tf.range(1,13,dtype=DTYPE), shape=(2,3,1,2))
    h = kepler(x)
    expected_h = tf.reshape([1/2 * (2**2 + 4**2 + 6**2) + 1./tf.sqrt(1. + 3**2 + 5**2),
                             1/2 * (8**2 + 10**2 + 12**2) + 1./tf.sqrt(7.**2 + 9**2 + 11**2)],
                             shape=(2,))
    assert_allclose(h, expected_h)
    print('test_kepler passed')
test_kepler()

def test_parameterized_neumann():
    # q=1,3; p=2;4
    x = tf.reshape(tf.range(1,5,dtype=DTYPE), shape=(1,2,1,2))
    ks = tf.constant([.1,.2], dtype=DTYPE)
    h = parameterized_neumann(ks)(x)
    expected_h = tf.constant(0.5*((1*4-3*2)**2) + \
                             0.5*(.1*1**2 + .2*3**2))
    assert_allclose(h, expected_h)
    # q=1,3,5; p=2;4;6
    # q=7,9,11; p=8;10;12
    x = tf.reshape(tf.range(1,13,dtype=DTYPE), shape=(2,3,1,2))
    ks = tf.constant([.1,.2,.3], dtype=DTYPE)
    h = parameterized_neumann(ks)(x)
    expected_h = tf.constant([0.5*((1*4-3*2)**2 + (1*6-5*2)**2 + (3*6-5*4)**2) + \
                              0.5*(.1*1**2 + .2*3**2 + .3*5**2),
                              0.5*((7*10-9*8)**2 + (7*12-11*8)**2 + (9*12-10*11)**2) + \
                              0.5*(.1*7**2 + .2*9**2 + .3*11**2)])
    assert_allclose(h, expected_h)
    print('test_parameterized_neumann passed')
test_parameterized_neumann()

def test_toy_hamiltonian():
    x = tf.constant([2.,3.], shape=(1,1,1,2))
    h = toy_hamiltonian(x)
    expected_h = tf.constant(1/2 * (2 - 1/4 * 3**2)**2 + 1/2 * 3**2 / 16)
    assert_equal(h, expected_h)
    print('test_toy_hamiltonian passed')
test_toy_hamiltonian()

# TODO: update
# def test_oscillator_hamiltonian():
#     x = tf.reshape(tf.range(1,5,dtype=DTYPE),[1,4,1]) # q = [1,2], p=[3,4]
#     h = oscillator_hamiltonian(x)
#     expected_h = tf.constant(1/2 * (9 + 16 + 1 + 4), dtype=DTYPE)
#     assert_equal(h, expected_h)
#     print('test_oscillator_hamiltonian passed')
# test_oscillator_hamiltonian()
#
# def test_diff_square_hamiltonian():
#     x = tf.reshape(tf.range(1,7,dtype=DTYPE),[1,6,1]) # q = [1,3,5], p=[2,4,6]
#     h = diff_square_hamiltonian(x)
#     # expected = 1/2 * (4 + 16 + 36 + (1.-3.)**2 + (3.-5.)**2 + (5.-1.)**2)
#     expected_h = tf.constant(1/2 * (4 + 16 + 36 + 4 + 4 + 16), dtype=DTYPE)
#     assert_equal(h, expected_h)
#     print('test_diff_square_hamiltonian passed')
# test_diff_square_hamiltonian()
#
# def test_fpu_alpha_hamiltonian():
#     x = tf.reshape(tf.range(1,7,dtype=DTYPE),[1,6,1]) # q = [1,3,5], p=[2,4,6]
#     h = fpu_alpha_hamiltonian(x)
#     expected_h = tf.constant(1/2 * (4 + 16 + 36 + 4 + 4 + 16) + 1/3 * (- 8 - 8 + 64), dtype=DTYPE)
#     assert_equal(h, expected_h)
#     print('test_fpu_alpha_hamiltonian passed')
# test_fpu_alpha_hamiltonian()
#
# def test_fpu_beta_hamiltonian():
#     x = tf.reshape(tf.range(1,7,dtype=DTYPE),[1,6,1]) # q = [1,3,5], p=[2,4,6]
#     h = fpu_beta_hamiltonian(x)
#     expected_h = tf.constant(1/2 * (4 + 16 + 36 + 4 + 4 + 16) + 1/4 * (16 + 16 + 4**4), dtype=DTYPE)
#     assert_equal(h, expected_h)
#     print('test_fpu_beta_hamiltonian passed')
# test_fpu_beta_hamiltonian()
#
# def test_toda_hamiltonian():
#     x = tf.reshape(tf.range(1,7,dtype=DTYPE),[1,6,1]) # q = [1,3,5], p=[2,4,6]
#     h = toda_hamiltonian(x)
#     expected_h = tf.constant(
#         1/2 * (4 + 16 + 36 + np.exp(1.-3.) + np.exp(3.-5.) + np.exp(5.-1.)),
#         dtype=DTYPE)
#     assert_equal(h, expected_h)
#     print('test_toda_hamiltonian passed')
# test_toda_hamiltonian()
