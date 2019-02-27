# Test functions which work only in graph mode - not eager.

import tensorflow as tf
import sys
sys.path.append("../")
from utils import *

DTYPE = tf.float32

def test_compute_gradK_penalty():
    z = tf.reshape(tf.range(8, dtype=DTYPE), shape=(2,1,2,2))
    K = 0.5 * tf.reduce_sum(tf.square(z), [1,2,3])
    actual = compute_gradK_penalty(K, z)
    # batch=0: q = 0,2 p = 1,3
    # batch=1: q = 4,6 p = 5,7
    expected = tf.reduce_mean(tf.constant([0.**2 + 2.**2 + (1.-1.)**2 + 3.**2,
                                           4.**2 + 6.**2 + (5.-1.)**2 + 7.**2]))
    with tf.Session() as sess:
        actual_np, expected_np = sess.run([actual, expected])
        assert (actual_np == expected_np).all()
    sess.close()
    print('test_compute_gradK_penalty passed')
test_compute_gradK_penalty()
