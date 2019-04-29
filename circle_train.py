import numpy as np
import tensorflow as tf

from inspect import getmembers, isfunction, isclass, getmodule
from tensorflow.contrib.training import HParams

from models import *
from hamiltonians import *
from losses import make_circle_loss
import hamiltonians
import losses
import data
from utils import make_train_op
from data import make_data


DTYPE=tf.float32
NP_DTYPE=np.float32

tf.set_random_seed(0)

FLAGS = tf.flags.FLAGS

# Hamiltonian flags
tf.flags.DEFINE_enum('hamiltonian', 'neumann_hamiltonian',
                     [f[0] for f in getmembers(hamiltonians, isfunction) if getmodule(f[1]) is hamiltonians],
                     'Hamiltonian function.')
tf.flags.DEFINE_integer("num_particles", 1, "Number of particles.")
tf.flags.DEFINE_integer("d", 3, "Space dimension.")

# Model flags
tf.flags.DEFINE_enum('base_dist', 'BaseDistributionActionAngle',
                     [f[0] for f in getmembers(data, isclass) if getmodule(f[1]) is data],
                     'Base distribution.')
tf.flags.DEFINE_enum('action_dist', 'dirac', ['dirac', 'exponential', 'normal'], 'Distribution of actions.')
tf.flags.DEFINE_integer("num_stacks_bijectors", 4, "Number of stacks of bijectors.")

# Training flags
tf.flags.DEFINE_string("logdir",
                       f"/tmp/logging/canonical_flows/{FLAGS.hamiltonian}_{FLAGS.loss}_{FLAGS.base_dist}",
                       "Directory to write logs.")
tf.flags.DEFINE_integer('dataset_size', 2**13, 'Set to float("inf") to keep sampling.')
tf.flags.DEFINE_integer('ckpt_freq', 1000, 'Checkpoint frequency')
tf.flags.DEFINE_boolean('visualize', True, 'Produce visualization.')
tf.flags.DEFINE_string("hparams", "", 'Comma separated list of "name=value" pairs e.g. "--hparams=learning_rate=0.3"')


def main(argv):

    hparams = HParams(minibatch_size=2**7, starter_learning_rate=0.00001, decay_lr="piecewise",
                      boundaries=[20000, 200000], boundary_values=[1e-5, 1e-6, 1e-6],
                      min_learning_rate=1e-6, grad_clip_norm=None)
    hparams.parse(FLAGS.hparams)

    z = make_data(hparams)

    # To account for periodicity start with oscillator flow
    stack = [OscillatorFlow()]
    for i in range(FLAGS.num_stacks_bijectors):
        stack.extend([ZeroCenter(),
                      LinearSymplecticTwoByTwo(),
                      SymplecticAdditiveCoupling(shift_model=IrrotationalMLP())])
        # SymplecticAdditiveCoupling(shift_model=MLP())])
    T = Chain(stack)

    with tf.name_scope("canonical_transformation"):
        x = T(z)
        if FLAGS.visualize:
            q,p = extract_q_p(x)
            tf.summary.histogram("q", q)
            tf.summary.histogram("p", p)
        K = eval(FLAGS.hamiltonian)(x)
        if FLAGS.visualize:
            tf.summary.histogram('K-Hamiltonian', K)

    loss = eval(FLAGS.loss)(K, z)
    tf.summary.scalar('loss', loss)

    step = tf.get_variable("global_step", [], tf.int64, tf.zeros_initializer(), trainable=False)
    train_op = make_train_op(hparams, loss, step)

    # Set the ZeroCenter bijectors to training mode:
    for i, bijector in enumerate(T.bijectors):
        if hasattr(bijector, 'is_training'):
            T.bijectors[i].is_training = True

    tf.contrib.training.train(train_op, logdir=FLAGS.logdir, save_checkpoint_secs=60)


if __name__ == '__main__':
    tf.app.run(main)