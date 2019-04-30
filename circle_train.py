import io
import numpy as np
import tensorflow as tf
import tfplot

from inspect import getmembers, isfunction, isclass, getmodule
from tensorflow.contrib.training import HParams

from models import *
from hamiltonians import *
from losses import make_circle_loss
import hamiltonians
import losses
import data
from utils import make_train_op
from data import make_trajectory


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
# tf.flags.DEFINE_enum('base_dist', 'BaseDistributionActionAngle',
#                      [f[0] for f in getmembers(data, isclass) if getmodule(f[1]) is data],
#                      'Base distribution.')
# tf.flags.DEFINE_enum('action_dist', 'dirac', ['dirac', 'exponential', 'normal'], 'Distribution of actions.')
tf.flags.DEFINE_integer('trajectory_duration', 2**10, 'Duration of trajectory.')
tf.flags.DEFINE_boolean('multiple_trajectories', True, 'Train over many different trajectories.')
tf.flags.DEFINE_integer("num_stacks_bijectors", 4, "Number of stacks of bijectors.")

# Training flags
tf.flags.DEFINE_integer('dataset_size', 2**13, 'Set to float("inf") to keep sampling.')
tf.flags.DEFINE_integer('ckpt_freq', 1000, 'Checkpoint frequency')
tf.flags.DEFINE_boolean('visualize', True, 'Produce visualization.')
tf.flags.DEFINE_string("hparams", "", 'Comma separated list of "name=value" pairs e.g. "--hparams=learning_rate=0.3"')
tf.flags.DEFINE_string("logdir", f"../logging/canonical_flows/{FLAGS.hamiltonian}", "Directory to write logs.")

def main(argv):

    hparams = HParams(minibatch_size=8, starter_learning_rate=0.00001, decay_lr="piecewise",
                      boundaries=[20000, 200000], boundary_values=[1e-5, 1e-6, 1e-6],
                      min_learning_rate=1e-6, grad_clip_norm=None)
    hparams.parse(FLAGS.hparams)

    hamiltonian_fn = eval(FLAGS.hamiltonian)
    traj = make_trajectory(hparams, hamiltonian_fn)
    # Shuffle along the time index
    traj = tf.random_shuffle(traj)

    stack = []
    for i in range(FLAGS.num_stacks_bijectors):
        stack.extend([ZeroCenter(),
                      LinearSymplectic(),
                      SymplecticAdditiveCoupling(shift_model=IrrotationalMLP())])
    T = Chain(stack)

    with tf.name_scope("canonical_transformation"):
        # traj is (num_time_samples,batch,d,n,2)
        num_time_samples = traj.shape[0]
        batch = traj.shape[1]
        # All the flows only allow for one batch index, so we reshape
        z = T.inverse(tf.reshape(traj, [num_time_samples * batch, FLAGS.d, FLAGS.num_particles, 2]))
        z = tf.reshape(z, [num_time_samples, batch, FLAGS.d, FLAGS.num_particles, 2])

    if FLAGS.visualize:
        # Add image summary. Must bring batch dimension to front
        qp_op = tfplot.wrap(qp_plot, name='PhasePlanes', batch=True)(tf.transpose(traj, [1, 0, 2, 3, 4]),
                                                                     tf.transpose(z, [1, 0, 2, 3, 4]))
        # qp_op = tf.expand_dims(qp_op, axis=0)  # Requires a batch dimension
        tf.summary.image("q-p", qp_op)

    loss = make_circle_loss(z, shift=-hparams.minibatch_size // 2)
    tf.summary.scalar('loss', loss)

    step = tf.get_variable("global_step", [], tf.int64, tf.zeros_initializer(), trainable=False)
    train_op = make_train_op(hparams, loss, step)

    tf.contrib.training.train(train_op, logdir=FLAGS.logdir, save_checkpoint_secs=60)


def qp_plot(qp, qp_hat):
    fig, ax = tfplot.subplots(FLAGS.num_particles, FLAGS.d, figsize=(12, 4))
    for dim in range(FLAGS.d):
        q, p = extract_q_p(qp)
        qhat, phat = extract_q_p(qp_hat)
        ax[dim].scatter(q[:, dim, 0, 0], p[:, dim, 0, 0])
        ax[dim].scatter(qhat[:, dim, 0, 0], phat[:, dim, 0, 0])

    return fig



if __name__ == '__main__':
    tf.app.run(main)