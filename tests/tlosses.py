"""
Tests for classes and functions in losses
"""

import tensorflow as tf
import sys
sys.path.append("../")
from losses import *
from utils import assert_equal, assert_allclose
import numpy as np

# Suppress the warning till they fix this:
# lib/python3.5/site-packages/tensorflow/python/util/tf_inspect.py:75:
# DeprecationWarning: inspect.getargspec() is deprecated, use inspect.signature() instead
import warnings
warnings.filterwarnings("ignore")

tf.enable_eager_execution()
DTYPE=tf.float32
NP_DTYPE=np.float32

tf.set_random_seed(0)

def test_make_circle_loss():
    d = 2
    n = 1
    N = 2
    b = 1
    tot = N*b*d*n*2
    # q[0,0]^2 = 0, q[0,1]^2 = 2
    # p[0,0]^2 = 1, p[0,1]^2 = 3
    # q[1,0]^2 = 4, q[1.1]^2 = 6
    # p[1,0]^2 = 5, p[1,1]^2 = 7

    # loss = (q[0,0]^2+p[0,0]^2 - (q[1,0]^2+p[1,0]^2))^2
    #      + (q[0,1]^2+p[0,1]^2 - (q[1,1]^2+p[1,1]^2))^2
    #      + (q[1,0]^2+p[1,0]^2 - (q[0,0]^2+p[0,0]^2))^2
    #      + (q[1,1]^2+p[1,1]^2 - (q[0,1]^2+p[0,1]^2))^2
    #      =2*(q[0,0]^2+p[0,0]^2 - (q[1,0]^2+p[1,0]^2))^2
    #      +2*(q[0,1]^2+p[0,1]^2 - (q[1,1]^2+p[1,1]^2))^2

    z = tf.sqrt( tf.reshape(tf.range(0,tot,dtype=DTYPE),(N,b,d,n,2)) )
    #      = 2*(0       +1        - (4       +5       ))^2
    #      + 2*(2       +3        - (6       +7       ))^2    
    expected_loss = tf.constant(2. * ((1-(4+5))**2 + (5-(6+7))**2), dtype=DTYPE)
    loss = make_circle_loss(z)
    assert_allclose(loss, expected_loss)
    print("test_make_circle_loss passed")
test_make_circle_loss()