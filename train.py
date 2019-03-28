import numpy as np
import tensorflow as tf


from models import *
from hamiltonians import parameterized_neumann
from utils import make_train_op
from losses import make_loss
from data import make_data


DTYPE=tf.float32
NP_DTYPE=np.float32

tf.set_random_seed(0)

import warnings
warnings.filterwarnings("ignore")

frequencies = [.1, .2, .3]
settings = {
    'frequencies': frequencies,
    'hamiltonian': parameterized_neumann(frequencies),
    'd': 3,                    # space dimension
    'num_particles': 1,        # number of particles
    'minibatch_size': 2**7,    # Mini batch size
    'dataset_size': 2**13,      # Set to float("inf") for keeping sampling.
    'num_stacks_bijectors': 4, # Number of bijectors
    'log_dir' : "/tmp/log/im_tests/neumann-3/with-grad-clip",
    'ckpt_freq': 1000,
    'train_iters': 2,
    'visualize': True,
#    'grad_clip_norm': 1e-10, # clip norm to val. Comment for no gradient clipping
    'starter_learning_rate': 0.00001,
    'decay_lr': "piecewise",
    'boundaries': [20000, 200000], # for piecewise decay
    'values': [1e-5, 1e-6, 1e-6],  # for piecewise decay
    'min_learning_rate': 1e-6,
#     'decay_steps': 25000,  # ignored if decay_lr False
#     'decay_rate': 0.5,     # ignored if decay_lr False (decayed_learning_rate = learning_rate *
#                            #                            decay_rate ^ (global_step / decay_steps))
    'loss': "dKdphi",
    'base_dist': "action_dirac_angle",
#    'value_actions': [0.1324, 0.0312, 0.2925],
#    'elastic_net_coeff': 1.
    }

# Choose a batch of actions: needs to be divisor of dataset_size or minibatch_size if infinite dataset
r = np.random.RandomState(seed=0)
num_samples_actions = 2 # number of distinct actions (Liouville torii)
sh = (num_samples_actions, settings['d'], settings['num_particles'], 1)
settings['value_actions'] = r.rand(*sh).astype(NP_DTYPE)



# To account for periodicity start with oscillator flow
stack = [OscillatorFlow()]
for i in range(settings['num_stacks_bijectors']):
    stack.extend([ZeroCenter(),
                  LinearSymplecticTwoByTwo(),
                  SymplecticAdditiveCoupling(shift_model=IrrotationalMLP())])
                  #SymplecticAdditiveCoupling(shift_model=MLP())])
T = Chain(stack)

step = tf.get_variable("global_step", [], tf.int64, tf.zeros_initializer(), trainable=False)

with tf.Session() as sess:

    z = make_data(settings, sess)

loss = make_loss(settings, T, z)

train_op = make_train_op(settings, loss, step)

# sess.run(tf.global_variables_initializer())


# Set the ZeroCenter bijectors to training mode:
for i, bijector in enumerate(T.bijectors):
    if hasattr(bijector, 'is_training'):
        T.bijectors[i].is_training = True

tf.contrib.training.train(train_op, logdir=settings['log_dir'], save_checkpoint_secs=60)

