"""
Tests for classes and functions in models
"""

import tensorflow as tf
from models import *
from utils import assert_equal, assert_allclose, run_eagerly, is_symplectic

DTYPE=tf.float32
# TODO: Set random number generator for reproducibility

# Class used in the test
class TimesTwoBijector(tf.keras.Model):
    """q,p -> 2*q,2*p"""
    def __init__(self):
        super(TimesTwoBijector, self).__init__()

    def call(self, x):
        return 2.*x

    def inverse(self, x):
        return x/2.

class TimesBijector(tf.keras.Model):
    """q,p -> 2*q,2*p"""
    def __init__(self, multiplier):
        super(TimesBijector, self).__init__()
        self.multiplier = multiplier

    def call(self, x):
        return self.multiplier*x

    def inverse(self, x):
        return x/self.multiplier

# Bijectors
@run_eagerly
def testNICE():
    x = tf.constant([1,2], shape=(1,2,1))
    model = NICE(tf.keras.layers.Lambda(lambda x: x))

    # Test call
    y = model(x)
    expected_y = tf.constant([1,2 + 1],shape=(1,2,1))
    assert_equal(y, expected_y)

    # Test inverse
    inverted_y = model.inverse(y)
    assert_equal(x, inverted_y)

@run_eagerly
def testSymplecticExchange():
    batch_size = 3
    phase_space_size = 2
    x = tf.ones([batch_size, phase_space_size, 1])

    # Test call
    model = SymplecticExchange()
    y = model(x)
    expected_y = tf.concat([tf.ones([batch_size, phase_space_size//2, 1]),
                           -tf.ones([batch_size, phase_space_size//2, 1])],
                           1)
    assert_equal(y, expected_y)

    # Test inverse
    inverted_y = model.inverse(y)
    assert_equal(x, inverted_y)

@run_eagerly
def testSqueezeAndShift():
    x = tf.constant([1.,2.], shape=(1,2,1))
    model = SqueezeAndShift(tf.keras.layers.Lambda(lambda x: x))

    # Test call
    y = model(x)
    expected_y = tf.constant([1.,2. + 1.],shape=(1,2,1))
    assert_equal(y, expected_y)

    # Test inverse
    inverted_y = model.inverse(y)
    assert_equal(x, inverted_y)

@run_eagerly
def testBijectorsAreSymplectic():
    phase_space_dim = 4

    bijectors = [NICE, SqueezeAndShift]
    for bijector in bijectors:

        model = bijector(shift_model=CNNShiftModel2())
        x = tf.random_normal((1, phase_space_dim, 1), dtype=DTYPE)
        assert(is_symplectic(model, x))

@run_eagerly
def testChain():
    batch_size = 3
    phase_space_size = 2
    x = tf.ones([batch_size, phase_space_size, 1])
    bijectors = [SymplecticExchange() for i in range(3)]

    # Test call
    model = Chain(bijectors)
    y = model(x)
    # q,p -> p,-q -> -q,-p -> -p,q
    expected_y = tf.concat([-tf.ones([batch_size, phase_space_size//2, 1]),
                           tf.ones([batch_size, phase_space_size//2, 1])],
                           1)
    assert_equal(y, expected_y)

    # Test inverse
    inverted_y = model.inverse(y)
    assert_equal(x, inverted_y)

# Neural networks
@run_eagerly
def testMLP():
    batch_size = 3
    input_size = 7
    input = tf.ones([batch_size, input_size, 1])

    # Test default: return_gradient and activation=elu.
    model = MLP()
    out = model(input)
    assert(out.shape == input.shape)

    # Test default: return_gradient = false
    model = MLP()
    out = model(input)
    assert(out.shape == input.shape)

@run_eagerly
def testCNNShiftModel():
    model = CNNShiftModel()
    x = tf.ones([4,2,3])
    y = model(x)
    assert(y.shape == x.shape)

# Architectures
@run_eagerly
def testMultiScaleArchitecture():
    # Test:
    arch = MultiScaleArchitecture( [TimesTwoBijector() for i in range(3)] )
    # batch = 1
    z = tf.ones([1,8,1], dtype=DTYPE)
    x = arch(z)
    expected_x = tf.constant([2,2,2,2,4,4,8,8], shape=(1,8,1), dtype=DTYPE)
    assert_equal(x, expected_x)    
    # Inverse
    assert_equal(z, arch.inverse(x))

    # Another test for inverse:
    batch = 64
    L = 4
    arch = MultiScaleArchitecture( [TimesTwoBijector() for i in range(L)] )
    phase_space_dim = 2**L # Minimum value that makes sense
    z = tf.ones([batch,phase_space_dim,1], dtype=DTYPE)
    assert_equal(z, arch.inverse(arch(z)))
    x = tf.ones([batch,phase_space_dim,1], dtype=DTYPE)
    assert_equal(x, arch(arch.inverse(x)))

# System tests
@run_eagerly
def systemTestChain():
    # Test real case scenario:
    num_steps_flow = 4
    phase_space_dim = 16
    batch = 64
    bijectors = [SqueezeAndShift(shift_model=CNNShiftModel()) if i % 2 == 0 
                 else SymplecticExchange() 
                 for i in range(num_steps_flow)]
    model = Chain(bijectors)    

    z = tf.ones([batch,phase_space_dim,1], dtype=DTYPE)
    assert_allclose(z, model.inverse(model(z)))
    x = tf.ones([batch,phase_space_dim,1], dtype=DTYPE)
    assert_allclose(z, model.inverse(model(z)))

@run_eagerly
def systemTestMultiScaleArchitecture():
    # Test real case scenario:
    num_scales = 4
    num_steps_flow = 4
    phase_space_dim = 2**num_scales
    batch = 4
    flows = []
    for i in range(num_scales):
        bijectors = [SqueezeAndShift(shift_model=CNNShiftModel()) if i % 2 == 0 
                     else SymplecticExchange() 
                     for i in range(num_steps_flow)]
        flows.append(Chain(bijectors))
    model = MultiScaleArchitecture(flows)
 
    tf.set_random_seed(0)
    z = tf.random_normal([batch,phase_space_dim,1], dtype=DTYPE)
    assert_allclose(z, model.inverse(model(z)))
    tf.set_random_seed(1)   
    x = tf.random_normal([batch,phase_space_dim,1], dtype=DTYPE)
    assert_allclose(x, model(model.inverse(x)))

