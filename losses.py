import tensorflow as tf
from utils import extract_q_p, join_q_p, normsq_nobatch
from models import MLPHamiltonian

def make_loss(settings, T, z, name):

    with tf.name_scope("canonical_transformation"):
        x = T(z)
        if settings['visualize']:
            tf.summary.histogram("x", x)
        K = settings['hamiltonian'](x)
        if settings['visualize']:
            tf.summary.histogram('K-Hamiltonian', K)

    with tf.name_scope("loss"):
        if name == "conserved_radii":
            # Action variables as radii
            q_prime, p_prime = extract_q_p(z)
            dq_prime, dp_prime = extract_q_p( tf.gradients(K, z)[0] )
            loss = tf.reduce_mean( tf.square(q_prime * dp_prime - p_prime * dq_prime) )

        elif name == "K_equal_F1_loss":
            # K = F_1
            _, F = extract_q_p(z)
            loss = tf.reduce_mean( tf.square(K - F[:,0,0,0]) )

        elif name == "K_equal_action_H":
            # K = NN(I). Use MLP Hamiltonian
            _, I = extract_q_p(z)
            action_H = MLPHamiltonian()
            H_I = action_H(I)
            # Add a residual part corresponding to GGE, so that H should be small
            H_I += tf.reduce_sum(I, [1,2,3])
            if settings['visualize']:
                tf.summary.histogram('action-Hamiltonian', H_I)
            loss = tf.reduce_mean( tf.square(K - H_I) )

        elif name == "K_indep_phi_T_periodic":
            # K independent of phi, T periodic
            phi, I = extract_q_p(z)
            dphi, _ = extract_q_p( tf.gradients(K, z)[0] )
            # Impose K = K(I): normsq(dphi). (Here dphi has shape [N,d,n,1])
            normsq_dphi = normsq_nobatch( dphi )
            # MC estimate of the integral over I,phi:
            loss = tf.reduce_mean( normsq_dphi )
            # Impose phi periodic over periods (truncate): sum_n normsq(T(phi,I) - T(phi+2*pi*n,I))
            periods = tf.constant([-1,1], dtype=DTYPE)
            periodic_constraint = tf.map_fn(lambda n : normsq_nobatch( x - T(join_q_p(phi + 2*np.pi*n, I)) ), periods)
            periodic_constraint = tf.reduce_sum(periodic_constraint, 0) # sum_n
            # MC estimate of the integral over I,phi:
            periodic_constraint = tf.reduce_mean( periodic_constraint )
            # Sum constraints
            multiplier = 1.
            loss += multiplier * periodic_constraint

        else:
            raise NameError('loss %s not implemented', name)

        tf.summary.scalar('loss', loss)
    return loss
