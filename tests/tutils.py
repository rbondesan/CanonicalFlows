"""
Tests for functions in utils
"""

import tensorflow as tf
import sys
sys.path.append("../")
from utils import *

DTYPE = tf.float32
tf.enable_eager_execution()

def test_extract_q_p():
    x = tf.constant([1,2,3,4],shape=(1,2,2))
    q, p = extract_q_p(x)
    expected_q = tf.constant([1,3],shape=(1,2,1))
    expected_p = tf.constant([2,4],shape=(1,2,1))
    assert_equal(q, expected_q)
    assert_equal(p, expected_p)
    # 2d
    x =  tf.reshape(tf.range(1,1+5*3*3*2),shape=(5,3,3,2))
    q, p = extract_q_p(x)
    expected_q = tf.reshape(tf.range(1,1+5*3*3*2,delta=2),shape=(5,3,3,1))
    expected_p = tf.reshape(tf.range(2,1+5*3*3*2,delta=2),shape=(5,3,3,1))
    assert_equal(q, expected_q)
    assert_equal(p, expected_p)
    print('test_extract_q_p passed')
test_extract_q_p()

def test_join_q_p():
    q = tf.constant([1,3],shape=(1,2,1))
    p = tf.constant([2,4],shape=(1,2,1))
    x = join_q_p(q, p)
    expected_x = tf.constant([1,2,3,4],shape=(1,2,2))
    assert_equal(x, expected_x)
    # Check that join_q_p inverts extract_q_p
    q = tf.constant(np.arange(0,300,2),shape=(6,5,5,1))
    p = tf.constant(np.arange(1,300,2),shape=(6,5,5,1))
    new_q, new_p = extract_q_p(join_q_p(q, p))
    assert_equal(q, new_q)
    assert_equal(p, new_p)
    print('test_join_q_p passed')
test_join_q_p()

def test_get_phase_space_dim():
    x = tf.reshape(tf.range(1,101),shape=(5,5,2,2))
    assert_equal(get_phase_space_dim(x.shape), tf.constant(20))
    print('test_get_phase_space_dim passed')
test_get_phase_space_dim()

# def test_indicator_fcn():
#     x = tf.reshape(tf.range(16), shape=(2,2,2,2))
#     low = 0
#     high = 13
#     expected = tf.constant(0.)
#     actual = indicator_fcn(low, high, x)
#     assert_equal(actual, expected)
#     print('test_indicator_fcn passed')
# test_indicator_fcn()

def test_confining_potential():
    x = tf.constant([-2., -1., -.23, .8, 1.5])
    low = -1.
    high = 1.
    expected = tf.constant([2.-1., 0., 0., 0., 1.5-1.])
    actual = confining_potential(x, low, high)
    assert_equal(actual, expected)
    #
    x = tf.constant([-3., 1.5])
    low = [-1., -10.]
    high = [1.,10.]
    expected = tf.constant([-(-3.+1.), 0.])
    actual = confining_potential(x, low, high)
    assert_equal(actual, expected)
    print('test_confining_potential passed')
test_confining_potential()

# TODO: update
# def test_split():
#     x=tf.constant([1,2,3,4], shape=(1,4,1)) # q=1,3; p=2,4
#     z1,z2=split(x)
#     expected_z1 = tf.constant([1,2], shape=(1,2,1))
#     expected_z2 = tf.constant([3,4], shape=(1,2,1))
#     assert_equal(z1, expected_z1)
#     assert_equal(z2, expected_z2)
#     print('test_split passed')
# test_split()

# TODO: update
# def test_lattice_shift():
#     x=tf.constant([1,2,3,4], shape=(1,2,2)) # q=1,3; p=2,4
#     x_shifted=tf.constant([3,4,1,2], shape=(1,2,2)) # q=3,1; p=4,2
#     assert_equal(lattice_shift(x), x_shifted)
#
#     # x[0,:,:]=[0,1,...,5], x[1,:,:]=[6,7,...,11]
#     x=tf.reshape(tf.range(12), [2,3,2])
#     q_shifted=tf.constant([[4,0,2],[10,6,8]], shape=(2,3,1))
#     p_shifted=tf.constant([[5,1,3],[11,7,9]], shape=(2,3,1))
#     assert_equal(lattice_shift(x), join_q_p(q_shifted,p_shifted))
#     print('test_lattice_shift passed')
# test_lattice_shift()

# TODO: fix
# def test_is_simplectic():
#     from models import SymplecticExchange
#     model = SymplecticExchange()
#     x = tf.reshape(tf.range(4, dtype=DTYPE), shape=(1,2,2))
#     assert(is_symplectic(model, x))
#     print('test_is_simplectic passed')
# test_is_simplectic()

def test_system_flow():
    # test values. Require T to have call and inverse. f to have
    # inverse and log_jacobian_det
    class StubT():
        def __call__(self, x):
            return x

        def inverse(self, x):
            # Use on purpose a wrong inverse since system flow should
            # not depend on it
            return x + 1

    class StubF():
        def inverse(self, x):
            return x - 1

        def log_jacobian_det(self, x):
            N = x.shape.as_list()[0]
            return tf.ones(N,)

    T = StubT()
    f = StubF()

    q0 = tf.constant([1,2], shape=[1,2,1,1], dtype=DTYPE)
    p0 = tf.constant([5,3], shape=[1,2,1,1], dtype=DTYPE)

    ts = tf.range(1,3, dtype=DTYPE)

    qs, ps, omega = system_flow(q0, p0, T, f, ts, prior='normal')

    # phi0 = tf.constant([2,3], shape=[1,2,1,1])
    # I0 = tf.constant([6,4], shape=[1,2,1,1])

    # omega(I):
    # u = tf.constant([5,3], shape=[1,2,1,1]); 1
    # g.gradient(minus_log_rho, I) = grad(..., u) = [u1 , u2] = [5, 3]

    # sh = [2,1,1,1]
    # Is = [ [6, 4]  # t = 1
    #        [6, 4]] # t = 2
    # phis = [ [2 + 5 * 1, 3 + 3 * 1]  # t = 1
    #          [2 + 5 * 2, 3 + 3 * 2]] # t = 2
    # same as qs, ps:
    expected_qs = tf.constant([[7,6],[12,9]], shape=[2,2,1,1], dtype=DTYPE)
    expected_ps = tf.constant([[6,4],[6,4]], shape=[2,2,1,1], dtype=DTYPE)
    assert_equal(qs, expected_qs)
    assert_equal(ps, expected_ps)
    print("test_system_flow passed")
test_system_flow()

def test_system_flow_iom():
    # test the case of prior = integrals_of_motion
    class StubT():
        def __call__(self, x):
            return x

        def inverse(self, x):
            # Use on purpose a wrong inverse since system flow should
            # not depend on it
            return x + 1

    class StubF():
        def __init__(self):
            self.log_scale = tf.ones([2,1,1])

    T = StubT()
    f = StubF()

    q0 = tf.constant([1,2], shape=[1,2,1,1], dtype=DTYPE)
    p0 = tf.constant([5,3], shape=[1,2,1,1], dtype=DTYPE)

    ts = tf.range(1,3, dtype=DTYPE)

    qs, ps, omega = system_flow(q0, p0, T, f, ts, prior='integrals_of_motion')

    # Q0 = tf.constant([2,3], shape=[1,2,1,1])
    # P0 = tf.constant([6,4], shape=[1,2,1,1])

    # omega = [exp(-1), 0]

    # sh = [2,1,1,1]
    # Ps = [ [6, 4]  # t = 1
    #        [6, 4]] # t = 2
    # Qs = [ [2 + exp(-1) * 1, 3]  # t = 1
    #        [2 + exp(-1) * 2, 3]] # t = 2
    # same as qs, ps:
    expected_qs = tf.constant([[2+np.exp(-1.),3],[2+np.exp(-1.)*2.,3]], shape=[2,2,1,1], dtype=DTYPE)
    expected_ps = tf.constant([[6,4],[6,4]], shape=[2,2,1,1], dtype=DTYPE)
    assert_equal(qs, expected_qs)
    assert_equal(ps, expected_ps)
    print("test_system_flow_iom passed")
test_system_flow_iom()
