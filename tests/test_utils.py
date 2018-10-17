"""
Tests for functions in utils
"""

import tensorflow as tf
from utils import *

DTYPE = tf.float32

@run_eagerly
def test_extract_q_p():
    x = tf.constant([1,2,3,4],shape=(1,4,1))
    q, p = extract_q_p(x)
    expected_q = tf.constant([1,3],shape=(1,2,1))
    expected_p = tf.constant([2,4],shape=(1,2,1))
    assert_equal(q, expected_q)
    assert_equal(p, expected_p)

@run_eagerly
def test_join_q_p():
    q = tf.constant([1,2],shape=(1,2,1))
    p = tf.constant([3,4],shape=(1,2,1))
    x = join_q_p(q, p)
    expected_x = tf.constant([1,3,2,4],shape=(1,4,1))
    assert_equal(x, expected_x)
    # Check that join_q_p inverts extract_q_p
    q = tf.constant(np.arange(0,100,2),shape=(5,10,1))
    p = tf.constant(np.arange(1,100,2),shape=(5,10,1))
    new_q, new_p = extract_q_p(join_q_p(q, p))
    assert_equal(q, new_q)
    assert_equal(p, new_p)
    #
    x = tf.constant(np.arange(100), shape=(5,20,1))
    q, p = extract_q_p(x)
    new_x = join_q_p(q, p)
    assert_equal(x, new_x)

@run_eagerly
def test_split():
    x=tf.constant([1,2,3,4], shape=(1,4,1)) # q=1,3; p=2,4
    z1,z2=split(x)
    expected_z1 = tf.constant([1,2], shape=(1,2,1))
    expected_z2 = tf.constant([3,4], shape=(1,2,1))
    assert_equal(z1, expected_z1)
    assert_equal(z2, expected_z2)

@run_eagerly
def test_lattice_shift():
    x=tf.constant([1,2,3,4], shape=(1,4,1)) # q=1,3; p=2,4
    x_shifted=tf.constant([3,4,1,2], shape=(1,4,1)) # q=3,1; p=4,2
    assert_equal(lattice_shift(x), x_shifted)

    # x[0,:,:]=[0,1,...,5], x[1,:,:]=[6,7,...,11]
    x=tf.reshape(tf.range(12), [2,6,1])
    x_shifted=tf.constant([ [[4,5,0,1,2,3]],[[10,11,6,7,8,9]] ], shape=(2,6,1))
    assert_equal(lattice_shift(x), x_shifted)

@run_eagerly
def test_is_symplectic():
    from models import SymplecticExchange
    model = SymplecticExchange()
    x = tf.reshape(tf.range(4, dtype=DTYPE), shape=(1,4,1))
    assert(is_symplectic(model, x))

# TODO: add system_flow test?
