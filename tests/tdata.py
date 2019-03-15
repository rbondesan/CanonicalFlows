"""
Tests for functions in data
"""

import tensorflow as tf
import sys
sys.path.append("../")
from data import *
from utils import assert_equal

DTYPE = tf.float32
tf.enable_eager_execution()

def test_DiracDistribution():
    # num_samples = 1
    sh = (3,2,1)
    vals = np.arange(3*2)
    d = DiracDistribution(sh, vals)
    x = d.sample(4)
    expected_shape = [4, 3, 2, 1]
    assert x.shape.as_list() == expected_shape
    assert_equal(x[0,...], x[1,...])
    assert_equal(x[1,...], x[2,...])
    assert_equal(x[2,...], x[3,...])
    # num_samples = 2
    sh = (3,1,1)
    vals = np.reshape(np.arange(2*3), (2,3))
    d = DiracDistribution(sh, vals)
    x = d.sample(4)
    expected_shape = [4, 3, 1, 1]
    assert x.shape.as_list() == expected_shape
    assert_equal(x[0,...], tf.reshape(tf.constant([0,1,2],dtype=DTYPE),(3,1,1)))
    assert_equal(x[1,...], tf.reshape(tf.constant([3,4,5],dtype=DTYPE),(3,1,1)))
    assert_equal(x[0,...], x[2,...])
    assert_equal(x[1,...], x[3,...])
    print('test_DiracDistribution passed')
test_DiracDistribution()

def testBaseDistributionActionAngle():
    settings = {'d': 2, 'num_particles': 3,'value_actions':np.arange(3*2)}
    # Default: exponential
    d = BaseDistributionActionAngle(settings)
    z = d.sample(15)
    expected_shape = [15, settings['d'], settings['num_particles'], 2]
    assert z.shape.as_list() == expected_shape
    # Default: Dirac
    d = BaseDistributionActionAngle(settings,action_dist='dirac')
    z = d.sample(15)
    expected_shape = [15, settings['d'], settings['num_particles'], 2]
    assert z.shape.as_list() == expected_shape
    print('testBaseDistributionActionAngle passed')
testBaseDistributionActionAngle()

def testBaseDistributionIntegralsOfMotion():
    settings = {'d': 2, 'num_particles': 3}
    d = BaseDistributionIntegralsOfMotion(settings)
    z = d.sample(15)
    expected_shape = [15, settings['d'], settings['num_particles'], 2]
    assert z.shape.as_list() == expected_shape
    print('testBaseDistributionIntegralsOfMotion passed')
testBaseDistributionIntegralsOfMotion()