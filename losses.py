import tensorflow as tf
from utils import extract_q_p, join_q_p, normsq_nobatch, compute_gradK_penalty
from utils import confining_potential
from models import MLPHamiltonian

FLAGS = tf.flags.FLAGS

## TODO: Remove. This was used before the call to make_loss
# Add ConstantShiftAndScale to enrich the base distribution which samples
# in the unit cube. Here we learn the scales (shift is canonical and
# already in ZeroCenter.)
# f = ConstantShiftAndScale(shift=False)
# unreg_loss = make_loss(settings, Chain([T,f]), z)
# tf.summary.scalar("unreg_loss", unreg_loss)
# # Add log det jac to prevent the scales to go to -infty to remove the
# # dependency on the variables from K
# regularization = -f.log_jacobian_det(z)
# tf.summary.scalar("regularization", regularization)
# loss = unreg_loss + regularization
# loss = make_loss(settings, Chain([T,f]), z)


def dKdphi(K, z):
    # K independent of phi
    dphi, _ = extract_q_p(tf.gradients(K, z)[0])
    if FLAGS.visualize:
        tf.summary.histogram('dKdphi', dphi)
    # loss = tf.reduce_mean( tf.square(dphi) + \
    #     settings['elastic_net_coeff'] * tf.pow( tf.abs(dphi), 3 ) )
    loss = tf.sqrt(tf.reduce_mean(0.5 * tf.square(dphi)))
    if 'lambda_range' in FLAGS:
        # With penalizing K (energy) outside low,high:
        range_reg = tf.reduce_mean(confining_potential(K, FLAGS.low_K_range, FLAGS.high_K_range))
        loss += FLAGS.lambda_range * range_reg
    if 'lambda_diff' in FLAGS:
        # add |K-val|^2 term
        diff_loss = tf.reduce_mean(tf.square(K - FLAGS.diff_val))
        loss += FLAGS.lambda_diff * diff_loss

    return loss

def conserved_radii(K, z):
    # Action variables as radii
    q_prime, p_prime = extract_q_p(z)
    dq_prime, dp_prime = extract_q_p(tf.gradients(K, z)[0])
    loss = tf.reduce_mean(tf.square(q_prime * dp_prime - p_prime * dq_prime))
    return loss

def K_equal_F1_loss(K, z):
    # K = F_1
    _, F = extract_q_p(z)
    loss = tf.reduce_mean(tf.square(K - F[:, 0, 0, 0]))
    return loss

def KL(K, z, T):
    # Here T can contain a non-symplectic part such as scaling.
    KL_loss = tf.reduce_mean(K - T.log_jacobian_det(z))
    # Gradient penalty regularization:
    gp = compute_gradK_penalty(K, z)
    # Range regularization
    range_reg = tf.reduce_mean(confining_potential(T(z), FLAGS.low_x_range, FLAGS.high_x_range))
    if FLAGS.visualize:
        tf.summary.scalar("KL_loss", KL_loss)
        # Monitor the derivative of K to understand how well we are doing
        # due to unknown Z in KL. Assume distribution propto e^-u_1.
        tf.summary.scalar("gradient_penalty", gp)
        tf.summary.scalar("range_reg", range_reg)
    loss = KL_loss + \
           FLAGS.lambda_gp * gp + \
           FLAGS.lambda_range * range_reg

    return loss

def K_equal_action_H(K, z):
    # K = NN(I). Use MLP Hamiltonian
    _, I = extract_q_p(z)
    action_H = MLPHamiltonian()
    H_I = action_H(I)
    # Add a residual part corresponding to GGE, so that H should be small
    # TODO: Need temperatures, otherwise, does not make too much sense,
    # think about Kepler problem, where Is are > 0 and H < 0.
    H_I += tf.reduce_sum(I, [1, 2, 3])
    if FLAGS.visualize:
        tf.summary.histogram('action-Hamiltonian', H_I)
    loss = tf.reduce_mean(tf.square(K - H_I))

    return loss


def K_indep_phi_T_periodic(K, z):
    # K independent of phi, T periodic
    phi, I = extract_q_p(z)
    dphi, _ = extract_q_p(tf.gradients(K, z)[0])
    # Impose K = K(I): normsq(dphi). (Here dphi has shape [N,d,n,1])
    normsq_dphi = normsq_nobatch(dphi)
    # MC estimate of the integral over I,phi:
    loss = tf.reduce_mean(normsq_dphi)
    # Impose phi periodic over periods (truncate): sum_n normsq(T(phi,I) - T(phi+2*pi*n,I))
    periods = tf.constant([-1, 1], dtype=DTYPE)
    periodic_constraint = tf.map_fn(lambda n: normsq_nobatch(x - T(join_q_p(phi + 2 * np.pi * n, I))), periods)
    periodic_constraint = tf.reduce_sum(periodic_constraint, 0)  # sum_n
    # MC estimate of the integral over I,phi:
    periodic_constraint = tf.reduce_mean(periodic_constraint)
    # Sum constraints
    multiplier = 1.
    loss += multiplier * periodic_constraint

    return loss


