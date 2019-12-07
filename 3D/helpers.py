import numpy as np
import tensorflow as tf
import sys
sys.path.append('../2D/')
from planar_kinematic_functions import normalize_tensor_angle

def quaternion_from_matrix(R):
    """ Transforms rotation matrices to quaternions.
        Params: R: 3x3- or nx3x3-matrix: rotation matrix (or matrices) to transform
        Returns: qs: nx4-matrix: the corresponding quaternions (real part first)
    """
    if len(R.shape) == 2:
        R = np.reshape(R, [-1,3,3])
    sess = tf.Session()
    with sess.as_default():
        R_T = tf.convert_to_tensor(R)
        qs = quaternion_from_matrix_tensor(R_T).eval()
    return qs
    
    
def quaternion_from_matrix_tensor(R):
    """ Transforms rotation matrices to quaternions using Tensorflow.
        Params: R: nx3x3-tensor: rotation matrix (or matrices) to transform
        Returns: qs: nx4-tensor: the corresponding quaternions (real part first)
    """
    epsilon = 1e-15 #small offset that guarantees no divisions by zero occur (and no square roots of negative numbers)
    max_value = 4  #the maximum value that 1 + trace(R) can reach
    
    #Source: http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    R = tf.cast(R, dtype=tf.float64)
    trace = tf.trace(R)
    
    radicand_g0 = tf.clip_by_value(1 + trace, epsilon, max_value)
    sqrt_trace_g0 = tf.sqrt(radicand_g0)
    q0_g0 = 0.5 * sqrt_trace_g0
    q1_g0 = 0.5 / sqrt_trace_g0 * (R[:,2,1] - R[:,1,2])
    q2_g0 = 0.5 / sqrt_trace_g0 * (R[:,0,2] - R[:,2,0])
    q3_g0 = 0.5 / sqrt_trace_g0 * (R[:,1,0] - R[:,0,1])
    
    radicand_c1 = tf.clip_by_value(1 + R[:,0,0] - R[:,1,1] - R[:,2,2], epsilon, max_value)
    sqrt_trace_l0_c1 = tf.sqrt(radicand_c1)
    q0_l0_c1 = 0.5 / sqrt_trace_l0_c1 * (R[:,2,1] - R[:,1,2])
    q1_l0_c1 = 0.5 * sqrt_trace_l0_c1
    q2_l0_c1 = 0.5 / sqrt_trace_l0_c1 * (R[:,1,0] + R[:,0,1])
    q3_l0_c1 = 0.5 / sqrt_trace_l0_c1 * (R[:,0,2] + R[:,2,0])
    
    radicand_c2 = tf.clip_by_value(1 + R[:,1,1] - R[:,0,0] - R[:,2,2], epsilon, max_value)
    sqrt_trace_l0_c2 = tf.sqrt(radicand_c2)
    q0_l0_c2 = 0.5 / sqrt_trace_l0_c2 * (R[:,0,2] - R[:,2,0])
    q1_l0_c2 = 0.5 / sqrt_trace_l0_c2 * (R[:,1,0] + R[:,0,1])
    q2_l0_c2 = 0.5 * sqrt_trace_l0_c2
    q3_l0_c2 = 0.5 / sqrt_trace_l0_c2 * (R[:,2,1] + R[:,1,2])
    
    radicand_else = tf.clip_by_value(1 + R[:,2,2] - R[:,0,0] - R[:,1,1], epsilon, max_value)
    sqrt_trace_l0_else = tf.sqrt(radicand_else)
    q0_l0_else = 0.5 / sqrt_trace_l0_else * (R[:,1,0] - R[:,0,1])
    q1_l0_else = 0.5 / sqrt_trace_l0_else * (R[:,0,2] + R[:,2,0])
    q2_l0_else = 0.5 / sqrt_trace_l0_else * (R[:,2,1] + R[:,1,2])
    q3_l0_else = 0.5 * sqrt_trace_l0_else
    
    trace_greater_0 = tf.greater(trace, 0)
    cond1 = tf.logical_and(tf.greater(R[:,0,0], R[:,1,1]), tf.greater(R[:,0,0], R[:,2,2]))
    cond2 = tf.greater(R[:,1,1], R[:,2,2])
    
    q0 = tf.where(trace_greater_0, q0_g0, tf.where(cond1, q0_l0_c1, tf.where(cond2, q0_l0_c2, q0_l0_else)))
    q1 = tf.where(trace_greater_0, q1_g0, tf.where(cond1, q1_l0_c1, tf.where(cond2, q1_l0_c2, q1_l0_else)))
    q2 = tf.where(trace_greater_0, q2_g0, tf.where(cond1, q2_l0_c1, tf.where(cond2, q2_l0_c2, q2_l0_else)))
    q3 = tf.where(trace_greater_0, q3_g0, tf.where(cond1, q3_l0_c1, tf.where(cond2, q3_l0_c2, q3_l0_else)))
    
    q01 = tf.stack([q0, q1], axis=-1)
    q23 = tf.stack([q2, q3], axis=-1)
    q = tf.concat([q01, q23], axis=-1)
    
    return q


def cartesian_error_sincos(robot, y_true, y_pred, w, joint_penalty, penalty_factor):
    """ This loss function transforms the predicted joint angles via the FK model
        into cartesian space where the error is calculated.
        Params:
            robot: object of the robot (e.g. class Compi) which contains the relevant robot parameters
                   and methods (e.g. FK)
            y_true: Tensor: The desired pose in cartesian space (position+quaternion vector)
            y_pred: Tensor: The predicted joint angles in joint space (in sin/cos representation)
            w: weight of the position error; weight of orientation error is (1-w)
            joint_penalty: Bool: add additional penalty term to loss in case of violations of joint limits
            penalty_factor: Scalar: defines how strong the joint limit violation penalty will be weighted
        Returns:
            loss: Loss in cartesian space. This is a weighted sum of the position loss
                  (mean euclidean distance) and the orientation loss (mean absolute orientation error).
                  If enabled, an extra amount of loss is added in case of joint limit violations.
    """
    
    y_pred_scaled = y_pred
    y_true = tf.cast(y_true, dtype=tf.float64)
        
    #decode the sin/cos joint prediction representation
    j = []
    for i in range(robot.n_DOFS):
        j.append(tf.atan2(y_pred_scaled[:,i], y_pred_scaled[:,i+robot.n_DOFS]))
        
    joints_pred = tf.cast(tf.stack(j, axis=-1), dtype=tf.float64)
    
    #propagate the predicted angles through the FK model
    SE3_pred = robot.FK_tensor(joints_pred)
        
    #convert the predicted SE3 matrices to position+quaternion
    pos_pred = SE3_pred[:,:3,3]
    rotation_pred = SE3_pred[:,:3,:3]
    quat_pred = quaternion_from_matrix_tensor(rotation_pred)
    
    #1. position loss (mean euclidean distance)
    loss_pos = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(pos_pred - y_true[:,:3]), axis=1)))
    #for mean squared error use instead:
    #loss_pos = tf.reduce_mean(tf.reduce_sum(tf.square(pos_pred - y_true[:,:3]), axis=1))
    
    #2. orientation loss (mean absolute orientation error)
    inner_products = tf.reduce_sum(tf.multiply(quat_pred, y_true[:,3:]), axis=1)
    abs_inner_products = tf.abs(inner_products)
    
    #make sure that rounding errors dont lead to nan values
    abs_inner_products = tf.clip_by_value(abs_inner_products, 0, 1)
    
    loss_ori = 2 * tf.reduce_mean(tf.acos(abs_inner_products))
    
    #3. weighted sum with hyperparameter w
    loss = w * loss_pos + (1-w) * loss_ori
    
    if joint_penalty:
        penalties_list = []
        for i, joint in enumerate(robot.sorted_joint_names):
            center = tf.convert_to_tensor(robot.joint_info[joint][4][1] + robot.joint_info[joint][4][0], dtype=tf.float64) / 2 #center of feasible space
            diff_to_center = normalize_tensor_angle(joints_pred[:,i] - center)
            
            #if diff_to_center is positive: actual angle is closer to upper joint limit
            #if diff_to_center is negative: actual angle is closer to lower joint limit
            greater_zero = tf.greater(diff_to_center, 0)
            diff_to_lower = normalize_tensor_angle(robot.joint_info[joint][4][0] - joints_pred[:,i])
            diff_to_upper = normalize_tensor_angle(joints_pred[:,i] - robot.joint_info[joint][4][1])
            relevant_diff = tf.where(greater_zero, diff_to_upper, diff_to_lower)
            
            #negative differences mean no violation of the joint limits
            #set them to zero so that they dont contribute to the loss
            violation_distances = tf.nn.relu(relevant_diff)
            
            #minimum distance from infeasible joint area to one of the limits is proportional to the loss.
            #this means, the larger the violation, i.e. the further away the predicted angle is from the joint limit,
            #the higher the loss. this way, the neural network can utilize the gradient to escape the infeasible area.
            penalties_list.append(violation_distances)
            
        penalties = tf.stack(penalties_list, axis=-1)
        penalty_sum = tf.reduce_sum(penalties)
        
        loss_with_penalty = loss + penalty_factor * (penalty_sum / tf.cast(tf.shape(joints_pred)[0], tf.float64))
        return tf.cast(loss_with_penalty, dtype=tf.float32)
    else:
        return tf.cast(loss, dtype=tf.float32)



def cartesian_loss_wrapper(robot, w, joint_penalty, penalty_factor):
    """ Wrapper function for the cartesian error loss function used for
        distal teaching.
    """
    def cart_loss(y_true, y_pred):
        return cartesian_error_sincos(robot, y_true, y_pred, w, joint_penalty, penalty_factor)
    return cart_loss



def halton(dim, n_sample):
    """Halton sequence.
    :param int dim: dimension
    :param int n_sample: number of samples.
    :return: sequence of Halton.
    :rtype: array_like (n_samples, n_features)
    """
    
    #source: https://gist.github.com/tupui/cea0a91cc127ea3890ac0f002f887bae
    
    def _primes_from_2_to(n):
        """Prime number from 2 to n.
        From `StackOverflow <https://stackoverflow.com/questions/2068372>`_.
        :param int n: sup bound with ``n >= 6``.
        :return: primes in 2 <= p < n.
        :rtype: list
        """
        sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool)
        for i in range(1, int(n ** 0.5) // 3 + 1):
            if sieve[i]:
                k = 3 * i + 1 | 1
                sieve[k * k // 3::2 * k] = False
                sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
        return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]
    
    def _van_der_corput(n_sample, base=2):
        """Van der Corput sequence.
        :param int n_sample: number of element of the sequence.
        :param int base: base of the sequence.
        :return: sequence of Van der Corput.
        :rtype: list (n_samples,)
        """
        n_sample, base = int(n_sample), int(base)
        sequence = []
        for i in range(n_sample):
            n_th_number, denom = 0., 1.
            while i > 0:
                i, remainder = divmod(i, base)
                denom *= base
                n_th_number += remainder / denom
            sequence.append(n_th_number)
    
        return sequence
    
    big_number = 10
    while 'Not enough primes':
        base = _primes_from_2_to(big_number)[:dim]
        if len(base) == dim:
            break
        big_number += 1000

    # Generate a sample using a Van der Corput sequence per dimension.
    sample = [_van_der_corput(n_sample + 1, dim) for dim in base]
    sample = np.stack(sample, axis=-1)[1:]

    return sample
    