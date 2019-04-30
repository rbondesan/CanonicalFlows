import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
from utils import join_q_p
from models import HamiltonianFlow

FLAGS = tf.flags.FLAGS

DTYPE = tf.float32
NP_DTYPE = np.float32
FLAGS = tf.flags.FLAGS


def make_trajectory(hparams, hamiltonian, start_point=None):
    # Use HamiltonianFlow as integrator of Hamiltonian
    integrator = HamiltonianFlow(hamiltonian,
                                 initial_t=0.,
                                 final_t=10.,
                                 num_steps=FLAGS.trajectory_duration)
    # Choose initial conditions at random... with gaussian the integration fails sometimes
    if FLAGS.resample_trajectories:
        x0 = tf.random.uniform([hparams.minibatch_size, FLAGS.d, FLAGS.num_particles, 2], minval=-1., maxval=1.)
        traj = integrator(x0, return_full_state=True)
    else:
        if start_point is None:
            x0 = 2 * np.random.rand(hparams.minibatch_size, FLAGS.d, FLAGS.num_particles, 2).astype(NP_DTYPE) - 1
        else:
            x0 = start_point
        with tf.Session() as sess:
            traj = sess.run(integrator(x0, return_full_state=True))

    # traj has shape (num_time_samples,batch,d,n,2)
    return traj


def make_data(hparams, value_actions=None):
    """Depending on the type of problem the return value is a tf.Tensor or a
    Dataset iterator of size minibatch."""

    with tf.name_scope("data"):

        if FLAGS.base_dist == 'BaseDistributionActionAngle':
            sampler = BaseDistributionActionAngle(action_dist=FLAGS.action_dist)
        elif FLAGS.base_dist == 'DiracDistribution':
            sampler = DiracDistribution(action_dist=value_actions)
        else:
            sampler = eval(FLAGS.base_dist)()

        # Create data: z is the minibatch
        if FLAGS.dataset_size == float("inf"):
            z = sampler.sample(hparams.minibatch_size)
        elif FLAGS.dataset_size == hparams.minibatch_size:
            with tf.Session() as sess:
                z = sess.run(sampler.sample(hparams.minibatch_size))
            z = tf.constant(z)
        else:
            # tf.data.Dataset. Create infinite dataset by repeating and
            # shuffling a finite number of samples.
            # Create in-memory data, assuming it fits...
            with tf.Session() as sess:
                Z = sess.run(sampler.sample(FLAGS.dataset_size))
            dataset = tf.data.Dataset.from_tensor_slices(Z.astype(NP_DTYPE))
            # repeat the dataset indefinetely
            dataset = dataset.repeat()
            # Shuffle data every epoch
            dataset = dataset.shuffle(buffer_size=Z.shape[0])
            # Specify maximum number of elements for prefetching.
            # dataset = dataset.prefetch(3 * settings['batch_size'])
            # Specify the minibatch size
            dataset = dataset.batch(hparams.minibatch_size)
            data_iterator = dataset.make_one_shot_iterator()
            z = data_iterator.get_next()

        ## This is for normal with annealing std dev:
        # Choose stddev =
        # 0.1 for step=0:10000
        # 0.5 for step=10000:50000
        # 1.0 for step>50000
        # boundaries = [2000, 10000]
        # values = [0.2, 0.3, 0.4]
        # stddev = tf.train.piecewise_constant(step, boundaries, values)
        # if settings['visualize']:
        #     tf.summary.scalar("stddev", stddev)
        # sh = [settings['minibatch_size'], settings['d'], settings['num_particles'], 2]
        # z = tf.random_normal(shape=sh, mean=0., stddev=stddev)

    return z


# Distributions
class DiracDistribution():
    def __init__(self, sh, value_actions):
        """value_actions is of shape (num_samples_actions, d, num_particles, 1)"""
        self.sh = sh
        # Choose arbitrary values. Set random seed for reproducibility.
#        r = np.random.RandomState(seed=0);
#        self.values = r.rand(*sh).astype(NP_DTYPE)

#        range = reduce(lambda x, y: x * y, sh)
#        self.values = tf.reshape( tf.range(1,range+1,dtype=DTYPE) / range, sh)
        
        value_actions = np.array(value_actions, dtype=NP_DTYPE)
        self.values = np.reshape(value_actions, sh)
        print("In DiracDistribution constructor: actions = ", self.values)

    def sample(self, N):
        assert N % self.sh[0] == 0, "N must be a multiple of num_samples_actions"
        return tf.tile(self.values, [N//self.sh[0], 1, 1, 1])


class BaseDistributionActionAngle():
    def __init__(self, action_dist='exponential'):
        sh = (FLAGS.d, FLAGS.num_particles, 1)
        # Actions
        if action_dist == 'exponential':
            self.base_dist_u = tfd.Independent(tfd.Exponential(rate=tf.ones(sh, DTYPE)),
                                               reinterpreted_batch_ndims=len(sh))
        elif action_dist == 'normal':
            self.base_dist_u = tfd.MultivariateNormalDiag(loc=tf.zeros(sh, DTYPE))
        elif action_dist == 'dirac':
            # Choose a batch of actions: needs to be divisor of dataset_size or minibatch_size if infinite dataset
            r = np.random.RandomState(seed=0)
            num_samples_actions = 2  # number of distinct actions (Liouville torii)
            u_shape = (num_samples_actions, FLAGS.d, FLAGS.num_particles, 1)
            value_actions = r.rand(*u_shape).astype(NP_DTYPE)
            self.base_dist_u = DiracDistribution(u_shape, value_actions)
        # Angles
        if 'high_phi' in FLAGS:
            high_phi = FLAGS.high_phi
        else:
            high_phi = 2*np.pi
        self.base_dist_phi = tfd.Independent(tfd.Uniform(low=tf.zeros(sh, DTYPE),
                                                         high=high_phi*tf.ones(sh, DTYPE)),
                                             reinterpreted_batch_ndims=len(sh))

    def sample(self, N):
        u = self.base_dist_u.sample(N)
        phi = self.base_dist_phi.sample(N)
        return join_q_p(phi, u)


class BaseDistributionNormal():
    def __init__(self):
        sh = [FLAGS.d, FLAGS.num_particles, 2]
        if 'truncated_range' in FLAGS:
            low = - FLAGS.truncated_range * tf.ones(sh, DTYPE)
            high = FLAGS.truncated_range * tf.ones(sh, DTYPE)
            self.base_dist_z = tfd.TruncatedNormal(loc=tf.zeros(sh, DTYPE),
                scale=1.0, low=low, high=high)
        else:
            self.base_dist_z = tfd.MultivariateNormalDiag(loc=tf.zeros(sh, DTYPE))

    def sample(self, N):
        return self.base_dist_z.sample(N)


class BaseDistributionIntegralsOfMotion():
    def __init__(self):
        """In the integrals of motion basis and their conjugate (F,psi), we
        can choose F_1 = H, so that the distribution is exponential for the first
        "momentum" variable and uniform for all the others. Here use standard
        normalization as the ranges will be learnt as part of the
        base-distribution-sampler part of the model"""
        # F
        self.base_dist_F1 = tfd.Exponential(rate=1.)
        sh = (FLAGS.d * FLAGS.num_particles - 1,)
        self.base_dist_otherF = tfd.Independent(tfd.Uniform(low=tf.zeros(sh, DTYPE),
                                                            high=tf.ones(sh, DTYPE)),
                                                reinterpreted_batch_ndims=len(sh))
        # Psi
        self.sh = [FLAGS.d, FLAGS.num_particles, 1]
        self.base_dist_Psi = tfd.Independent(tfd.Uniform(low=tf.zeros(self.sh, DTYPE),
                                                         high=tf.ones(self.sh, DTYPE)),
                                             reinterpreted_batch_ndims=len(self.sh))

    def sample(self, N):
        F1 = tf.reshape(self.base_dist_F1.sample(N), shape=[N,1]) # sh = [N,1]
        otherF = self.base_dist_otherF.sample(N)  # sh = [N,d*n-1]
        F = tf.concat([F1, otherF], 1)       # sh = [N,d*n]
        F = tf.reshape(F, shape=[N]+self.sh) # sh = [N,d,n,1]
        Psi = self.base_dist_Psi.sample(N)
        return join_q_p(Psi, F)
