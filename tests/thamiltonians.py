"""
Tests for functions in hamiltonians
"""

import tensorflow as tf
import sys
sys.path.append("../")
from hamiltonians import *
from utils import assert_equal, assert_allclose, join_q_p

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

def test_free():
    # q=1,3,5; p=2;4;6
    # q=7,9,11; p=8;10;12
    x = tf.reshape(tf.range(1,13,dtype=DTYPE), shape=(2,3,1,2))
    h = free(x)
    expected_h = tf.reshape([1/2 * (2**2 + 4**2 + 6**2),
                             1/2 * (8**2 + 10**2 + 12**2)],
                             shape=(2,))
    assert_allclose(h, expected_h)
    print('test_free passed')
test_free()

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

def test_open_toda():
    x = tf.reshape(tf.range(1,7,dtype=DTYPE),[1,1,3,2]) # q = [1,3,5], p=[2,4,6]
    h = open_toda(x)
    expected_h = tf.constant(
        1/2 * (4 + 16 + 36) +  np.exp(1.-3.) + np.exp(3.-5.),
        dtype=DTYPE)
    assert_equal(h, expected_h)
    print('test_open_toda passed')
test_open_toda()

def test_closed_toda():
    x = tf.reshape(tf.range(1,7,dtype=DTYPE),[1,1,3,2]) # q = [1,3,5], p=[2,4,6]
    h = closed_toda(x)
    expected_h = tf.constant(
        1/2 * (4 + 16 + 36) + np.exp(1.-3.) + np.exp(3.-5.) + np.exp(5.-1.),
        dtype=DTYPE)
    assert_allclose(h, expected_h)
    print('test_closed_toda passed')
test_closed_toda()

def test_closed_toda_3():
    # Choose points such that q1+q2+q3 = p1+p2+p3 = 0
    q = tf.reshape([.1, .2, -.3], [1,1,3,1])
    p = tf.reshape([.3, .5, -.8], [1,1,3,1])
    h1 = closed_toda( join_q_p(q,p) )
    # Fourier modes:
    # a1 = x + i y
    # = 1/sqrt(3)(1/w .1 + w .2 - .3) =
    # = 1/sqrt(3)((-1/2 - i sqrt(3)/2) .1 + (-1/2 + i sqrt(3)/2) .2 - .3)
    x = 1/tf.sqrt(3.) * ( -1/2. * (.1 + .2) - .3 )
    # y = 1/tf.sqrt(3.)( tf.sqrt(3.)/2. * (.1 - .2) )
    y = 1/2. * ( -.1 + .2 )
    # b1 = px + i py = 1/sqrt(3)(1/w p1 + w p2 + p3)
    # = 1/sqrt(3)( (-1/2 - i sqrt(3)/2) .3 + (1/2 + i sqrt(3)/2) .5 - .8)
    # = 1/sqrt(3)( -1/2 * (.3 + .5) - .8) - i 1/2 (.3 - .5)
    px = 1/tf.sqrt(3.) * ( -1/2. * (.3 + .5) - .8)
    py = -1/2. * (.3 - .5)
    kin = px**2 + py**2
    expected_kin = tf.reduce_sum(tf.square(p)) * .5
    assert_allclose(kin, expected_kin)
    q1 = q[0,0,0,0]
    q2 = q[0,0,1,0]
    q3 = q[0,0,2,0]
    expected_q12 = -2*y
    expected_q23 = y-tf.sqrt(3.)*x
    expected_q31 = y+tf.sqrt(3.)*x
    assert_allclose(q1-q2, expected_q12)
    assert_allclose(q2-q3, expected_q23)
    assert_allclose(q3-q1, expected_q31)
    h2 = kin + tf.exp(expected_q12) + tf.exp(expected_q23) + tf.exp(expected_q31)
    assert_allclose(h1, h2)
    print('test_closed_toda_3 passed')
test_closed_toda_3()

def test_calogero_moser():
    x = tf.reshape(tf.range(1,7,dtype=DTYPE),[1,1,3,2]) # q = [1,3,5], p=[2,4,6]
    h = calogero_moser(x, 'rational')
    expected_h = tf.constant(
        1/2 * (4 + 16 + 36 + 1 + 9 + 25) + 1./(1 - 3)**2 + 1./(1 - 5)**2 + 1./(3 - 5)**2,
        dtype=DTYPE)
    assert_allclose(h, expected_h)

    x = tf.reshape([-.1,1.,3.2,-.2,1.34,3.],[1,1,3,2]) # q = [-.1,3.2,1.34], p=[1.,-.2,3.]
    h = calogero_moser(x, 'rational', omegasq=.3)
    expected_h = tf.constant(
        1/2 * (1.**2 + .2**2 + 3.**2 + .3 * (.1**2 + 3.2**2 + 1.34**2)) + \
            1./(-.1 - 3.2)**2 + 1./(-.1 - 1.34)**2 + 1./(3.2 - 1.34)**2,
        dtype=DTYPE)
    assert_allclose(h, expected_h)    
    print('test_calogero_moser passed')
test_calogero_moser()