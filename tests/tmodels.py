"""
Tests for classes and functions in models
"""

import tensorflow as tf
import sys
sys.path.append("../")
from models import *
from utils import assert_equal, assert_allclose, is_symplectic

# Suppress the warning till they fix this:
# lib/python3.5/site-packages/tensorflow/python/util/tf_inspect.py:75:
# DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead
import warnings
warnings.filterwarnings("ignore")

tf.enable_eager_execution()
DTYPE=tf.float32

tf.set_random_seed(0)

# NormalizingFlow
def testAngleInvariantFlow():
    x = tf.constant([1,2], shape=(1,1,2))
    model = AngleInvariantFlow(TimesTwoBijector())
    # call
    expected_y = tf.constant([1,4], shape=(1,1,2))
    assert_equal(model(x), expected_y)
    print('testAngleInvariantFlow passed')
testAngleInvariantFlow()

def testSymplecticAdditiveCoupling():
    x = tf.constant([1,2], shape=(1,1,2))
    model = SymplecticAdditiveCoupling(tf.keras.layers.Lambda(lambda x: x))
    # Test call
    y = model(x)
    expected_y = tf.constant([1,2 + 1],shape=(1,1,2))
    assert_equal(y, expected_y)
    # Test inverse
    inverted_y = model.inverse(y)
    assert_equal(x, inverted_y)
    # Test zero log jacobian determinant
    z = tf.reshape(tf.range(1, 11), shape=(5,1,2))
    model = SymplecticAdditiveCoupling(tf.keras.layers.Lambda(lambda x: x))
    assert_equal(model.log_jacobian_det(z), tf.zeros([5], dtype=tf.int32))
    print('testSymplecticAdditiveCoupling passed')
testSymplecticAdditiveCoupling()

def testConstantShiftAndScale():
    z = tf.constant([.1,2,3,.4], shape=(2,2,1,1), dtype=DTYPE)
    model = ConstantShiftAndScale()
    # Test call
    x = model(z)
    assert_equal(x, z) # Since scale and shift are zero, identity
    assert_equal(model.inverse(x), z)
    # This is not robust as tf equal zeros is always true
    assert_equal(model.log_jacobian_det(z), tf.zeros((2,), dtype=DTYPE))
    print("testConstantShiftAndScale passed")
testConstantShiftAndScale()

def testPermute():
    z = tf.reshape(tf.range(0,6, dtype=DTYPE), shape=(2,3,1,1))
    # z[0,:] = [0,1,2], z[1,:] = [3,4,5]
    dim_to_split = 1; na = 1; nb = 3 - na
    model = Permute(dim_to_split, [na, nb])
    # Test call
    x = model(z)
    expected_x = tf.constant([1.,2.,0.,
                             4.,5.,3.], shape=(2,3,1,1))
    assert_allclose(x, expected_x)
    assert_allclose(model.log_jacobian_det(z), tf.zeros((2,), dtype=DTYPE))
    print("testPermute passed")
testPermute()

def testOscillatorFlow():
    # phi[0] = 0, I[0] = .5; phi[1] = 1; I[1] = 1.5
    z = tf.reshape(tf.range(0,4,dtype=DTYPE),[2,1,1,2]) / 2.
    expected_x = tf.reshape([tf.sqrt(2 * .5) * tf.sin(0.), tf.sqrt(2 * .5) * tf.cos(0.),
                             tf.sqrt(2 * 1.5) * tf.sin(1.), tf.sqrt(2 * 1.5) * tf.cos(1.) ],
                             [2,1,1,2])
    m = OscillatorFlow()
    assert_allclose(m(z), expected_x)
    assert_allclose(m.inverse(m(z)), z)
    #
    z = tf.reshape(tf.range(0,36,dtype=DTYPE),[3,2,3,2]) / 36.
    m = OscillatorFlow()
    assert_allclose(m.inverse(m(z)), z)
    print("testOscillatorFlow passed")
testOscillatorFlow()

def testAffineCoupling():
    z = tf.constant([.1,2,3,.4], shape=(2,2,1,1), dtype=DTYPE) # u[0,:]=.1,2, u[1,:]=3,.4
    model = AffineCoupling(ReplicateAlongChannels(), 1, [1,1])
    # Test call
    x = model(z)
    expected_x = tf.constant([np.exp(2) * .1 + 2, 2,
                              np.exp(.4) * 3 + .4, .4],shape=(2,2,1,1),
                              dtype=DTYPE)
    assert_allclose(x, expected_x)
    # Test inverse
    inverted_x = model.inverse(x)
    assert_allclose(z, inverted_x)
    # Test log jacobian determinant
    assert_allclose(model.log_jacobian_det(z), tf.constant([2, .4], dtype=DTYPE))
    #
    z = tf.constant([.1,2,3,.4,5,.6], shape=(2,3,1,1), dtype=DTYPE)
    # u[0,:]=.1,2,3, so ua = .1, ub = 2,3
    # u[1,:]=.4,5,.6, so ua = .4, ub = 5,.6
    model = AffineCoupling(ReplicateAlongChannels(truncate=True), 1, [1,2])
    # Test call
    x = model(z)
    expected_x = tf.constant([np.exp(2) * .1 + 2, 2, 3,
                              np.exp(5) * .4 + 5, 5, .6],shape=(2,3,1,1),
                              dtype=DTYPE)
    assert_allclose(x, expected_x)
    # Test inverse
    inverted_x = model.inverse(x)
    assert_allclose(z, inverted_x)
    # Test log jacobian determinant
    assert_allclose(model.log_jacobian_det(z), tf.constant([2, 5], dtype=DTYPE))
    print('testAffineCoupling passed')
testAffineCoupling()

def testAdditiveCoupling():
    z = tf.constant([.1,2,3,.4], shape=(2,2,1,1), dtype=DTYPE) # u[0,:]=.1,2, u[1,:]=3,.4
    val = 1
    out_shape = (2,1,1,1)
    model = AdditiveCoupling(ConstantNN(out_shape, val), 1, [1,1])
    # Test call
    x = model(z)
    expected_x = tf.constant([.1 + 1, 2,
                              3 + 1, .4],shape=(2,2,1,1),
                              dtype=DTYPE)
    assert_allclose(x, expected_x)
    # Test inverse
    inverted_x = model.inverse(x)
    assert_allclose(z, inverted_x)
    # Test log jacobian determinant
    assert_allclose(model.log_jacobian_det(z), tf.zeros([2], dtype=DTYPE))
    #
    z = tf.constant([.1,2,3,.4,5,.6], shape=(2,3,1,1), dtype=DTYPE)
    # u[0,:]=.1,2,3, so ua = .1, ub = 2,3
    # u[1,:]=.4,5,.6, so ua = .4, ub = 5,.6
    val = .123
    out_shape = (2,1,1,1) # shape of za
    model = AdditiveCoupling(ConstantNN(out_shape, val), 1, [1,2])
    # Test call
    x = model(z)
    expected_x = tf.constant([.1 + .123, 2, 3,
                              .4 + .123, 5, .6],shape=(2,3,1,1),
                              dtype=DTYPE)
    assert_allclose(x, expected_x)
    # Test inverse
    inverted_x = model.inverse(x)
    assert_allclose(z, inverted_x)
    # Test log jacobian determinant
    assert_allclose(model.log_jacobian_det(z), tf.zeros([2], dtype=DTYPE))
    # test positive shift
    z = tf.constant([.1,2,3,.4,5,.6], shape=(2,3,1,1), dtype=DTYPE)
    val = -.123
    out_shape = (2,1,1,1) # shape of za
    model = AdditiveCoupling(ConstantNN(out_shape, val), 1, [1,2], is_positive_shift=True)
    # Test call
    x = model(z)
    expected_x = tf.constant([.1 + .123, 2, 3,
                              .4 + .123, 5, .6],shape=(2,3,1,1),
                              dtype=DTYPE)
    assert_allclose(x, expected_x)
    # Test inverse
    inverted_x = model.inverse(x)
    assert_allclose(z, inverted_x)
    print('testAdditiveCoupling passed')
testAdditiveCoupling()

def testSymplecticExchange():
    batch_size = 3
    x = tf.ones([batch_size, 1, 2])
    # Test call
    model = SymplecticExchange()
    y = model(x)
    expected_y = tf.concat([tf.ones([batch_size, 1, 1]),
                           -tf.ones([batch_size, 1, 1])],
                           2)
    assert_equal(y, expected_y)
    # Test inverse
    inverted_y = model.inverse(y)
    assert_equal(x, inverted_y)
    # Test zero log jacobian determinant
    z = tf.reshape(tf.range(1, 11), shape=(5,1,2))
    assert_equal(model.log_jacobian_det(z), tf.zeros([5], dtype=tf.int32))
    # Assert symplecticity
    model = SymplecticExchange()
    assert( is_symplectic(model,
                          tf.reshape(tf.range(0,20,dtype=DTYPE),[1,2,5,2]) ) )
    print('testSymplecticExchange passed')
testSymplecticExchange()

def testSqueezeAndShift():
    x = tf.constant([1.,2.], shape=(1,1,2))
    model = SqueezeAndShift(tf.keras.layers.Lambda(lambda x: x))
    # Test call
    y = model(x)
    expected_y = tf.constant([1.,2. + 1.],shape=(1,1,2))
    assert_equal(y, expected_y)
    # Test inverse
    inverted_y = model.inverse(y)
    assert_equal(x, inverted_y)
    # Test zero log jacobian determinant
    z = tf.reshape(tf.range(1, 11), shape=(5,1,2))
    assert_equal(model.log_jacobian_det(z), tf.zeros([5], dtype=tf.int32))
    print('testSqueezeAndShift passed')
testSqueezeAndShift()

def testLinearSymplectic():
    # 2 samples, 5 particles in 2d. squeezing factors = [1,1]
    x = tf.reshape(tf.range(0,40,dtype=DTYPE), shape=(2,2,5,2))
    model = LinearSymplectic()
    y = model(x)
    assert(x.shape == y.shape)
    assert_allclose(model.inverse(y), x)
    model = LinearSymplectic()
    assert( is_symplectic(model,
                      tf.reshape(tf.range(0,20,dtype=DTYPE),[1,2,5,2]) ) )

    # 5 samples, 2 particles in 2d. squeezing factors = [2,2]
    # TODO
    print('testLinearSymplectic passed')
testLinearSymplectic()

# Neural networks
def testMLP():
    input = tf.ones([3, 2, 3, 1])
    # Test default: mode=symplectic_shift and activation=softplus
    model = MLP()
    out = model(input)
    assert(out.shape == input.shape)
    # We should be able to pass different minibatches
    input = tf.ones([15, 2, 3, 1])
    out = model(input)
    assert(out.shape == input.shape)
    #
    input = tf.ones([3, 5, 6, 7, 1])
    # Test default: mode=symplectic_shift and activation=softplus
    model = MLP()
    out = model(input)
    assert(out.shape == input.shape)
    # Test: mode=shift and activation=softplus
    model = MLP(mode="shift", activation="relu")
    out = model(input)
    assert(out.shape == input.shape)
    # Test: mode=shift_and_scale and activation=softplus
    model = MLP(mode="shift_and_scale", activation="relu")
    expected_shape = tf.constant([3, 5, 6, 7, 2])
    assert_equal(tf.shape(model(input)), expected_shape)
    # with out_dim
    model = MLP(mode="shift_and_scale", activation="relu", out_shape=[3,3,2,1,32])
    expected_shape = tf.constant([3, 3, 2, 1, 32])
    assert_equal(tf.shape(model(input)), expected_shape)
    print('testMLP passed')
testMLP()

# TODO: update
# def testCNNShiftModel():
#     model = CNNShiftModel()
#     x = tf.ones([4,2,3])
#     y = model(x)
#     assert(y.shape == x.shape)
#     print('testCNNShiftModel passed')
# testCNNShiftModel()

# Architectures
def testChain():
    batch_size = 3
    x = tf.ones([batch_size, 1, 2])
    bijectors = [SymplecticExchange() for i in range(3)]
    # Test call
    model = Chain(bijectors)
    y = model(x)
    # q,p -> p,-q -> -q,-p -> -p,q
    expected_y = tf.concat([-tf.ones([batch_size, 1, 1]),
                           tf.ones([batch_size, 1, 1])],
                           2)
    assert_equal(y, expected_y)
    # Test inverse
    inverted_y = model.inverse(y)
    assert_equal(x, inverted_y)
    # Test log jacobian determinant
    z = tf.ones([3, 2, 7, 1]) * 1.2345 # arbitrary
    n_models = 15
    model = Chain( [TimesTwoBijector() for i in range(n_models)] )
    expected_log_jac_det = n_models * tf.log(2.) * 2 * 7 * 1 * tf.ones((3,),dtype=tf.float32)
    assert_allclose(model.log_jacobian_det(z), expected_log_jac_det)
    print('testChain passed')
testChain()

# TODO: update
# def testMultiScaleArchitecture():
#     # Test:
#     arch = MultiScaleArchitecture( [TimesTwoBijector() for i in range(3)] )
#     # batch = 1
#     z = tf.ones([1,8,1], dtype=DTYPE)
#     x = arch(z)
#     expected_x = tf.constant([2,2,2,2,4,4,8,8], shape=(1,8,1), dtype=DTYPE)
#     assert_equal(x, expected_x)
#     # Inverse
#     assert_equal(z, arch.inverse(x))
#
#     # Another test for inverse:
#     batch = 64
#     L = 4
#     arch = MultiScaleArchitecture( [TimesTwoBijector() for i in range(L)] )
#     phase_space_dim = 2**L # Minimum value that makes sense
#     z = tf.ones([batch,phase_space_dim,1], dtype=DTYPE)
#     assert_equal(z, arch.inverse(arch(z)))
#     x = tf.ones([batch,phase_space_dim,1], dtype=DTYPE)
#     assert_equal(x, arch(arch.inverse(x)))
#     print('testMultiScaleArchitecture passed')
# testMultiScaleArchitecture()

# System tests
def systemTestAffineCouplingMLP():
    # Test real case scenario:
    batch = 4
    d = 3
    n = 2
    z = tf.ones([batch,d,n,1], dtype=DTYPE)
    model = AffineCoupling(shift_and_scale_model = MLP(activation=tf.nn.relu,
                                                       mode="shift_and_scale"),
                                                       split_dim = 2,
                                                       split_sizes = [1, 1])
    x = model(z)
    assert_equal(x.shape, z.shape)
    # Another test with non-even splitting
    split_dim = 1
    na = 1
    nb = 2
    model = AffineCoupling(shift_and_scale_model = MLP(activation=tf.nn.relu,
                                                       mode="shift_and_scale",
                                                       out_shape=[batch,na,n,2]),
                                                       split_dim = split_dim,
                                                       split_sizes = [na, nb])
    x = model(z)
    assert_equal(x.shape, z.shape)
    print('systemTestAffineCouplingMLP passed')
systemTestAffineCouplingMLP()


# TODO: update
# def systemTestChain():
#     # Test real case scenario:
#     num_steps_flow = 4
#     phase_space_dim = 16
#     batch = 64
#     bijectors = [SqueezeAndShift(shift_model=CNNShiftModel()) if i % 2 == 0
#                  else SymplecticExchange()
#                  for i in range(num_steps_flow)]
#     model = Chain(bijectors)
#
#     z = tf.ones([batch,phase_space_dim,1], dtype=DTYPE)
#     assert_allclose(z, model.inverse(model(z)))
#     x = tf.ones([batch,phase_space_dim,1], dtype=DTYPE)
#     assert_allclose(z, model.inverse(model(z)))
#     print('systemTestChain passed')
# systemTestChain()
#
# def systemTestMultiScaleArchitecture():
#     # Test real case scenario:
#     num_scales = 4
#     num_steps_flow = 4
#     phase_space_dim = 2**num_scales
#     batch = 4
#     flows = []
#     for i in range(num_scales):
#         bijectors = [SqueezeAndShift(shift_model=CNNShiftModel()) if i % 2 == 0
#                      else SymplecticExchange()
#                      for i in range(num_steps_flow)]
#         flows.append(Chain(bijectors))
#     model = MultiScaleArchitecture(flows)
#
#     tf.set_random_seed(0)
#     z = tf.random_normal([batch,phase_space_dim,1], dtype=DTYPE)
#     assert_allclose(z, model.inverse(model(z)))
#     tf.set_random_seed(1)
#     x = tf.random_normal([batch,phase_space_dim,1], dtype=DTYPE)
#     assert_allclose(x, model(model.inverse(x)))
#     print('systemTestMultiScaleArchitecture passed')
# systemTestMultiScaleArchitecture()
