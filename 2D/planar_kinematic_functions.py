import numpy as np
import math
from matplotlib import pyplot as plt
import tensorflow as tf
from timeit import default_timer as timer


def FK_2D(theta, link_lengths):
    """ 2D Forward Kinematic Model (revolute joints only):
        Transforms configurations into 2D cartesian space.
        Params:
            theta: nxk-matrix: Angles of the joints (rad)
                   n: number of samples
                   k: number of joints
            link_lengths: kx1-vector: lengths of the links
                          k: number of joints (=number of links)
        Returns:
            X: nx3-matrix:
               X[:,:2]: position x,y (same unit as link_lengths)
               X[:,2]:  orientation (rad)
    """
    theta = np.atleast_2d(theta)
    link_lengths = np.array(link_lengths)
    n_DOFS = theta.shape[1]
    assert n_DOFS == link_lengths.shape[0], "Numbers of joint angles and link lengths have to be equal"
    n_samples = theta.shape[0]
    
    X = np.zeros((n_samples,3))
    for j in range(n_DOFS):
        X[:,0] += np.cos(np.sum(theta[:,:j+1], axis=1))*link_lengths[j]
        X[:,1] += np.sin(np.sum(theta[:,:j+1], axis=1))*link_lengths[j]
    #normalize orientation to [-Pi, Pi]
    alpha = np.sum(theta, axis=1) % (2*math.pi)
    alpha[alpha > math.pi] -= 2*math.pi        
    X[:,2] = alpha
    return X


def IK_2D_3DOF(poses, link_lengths):
    """ 2D analytical Inverse Kinematic Model for a 3 DOF manipulator (revolute joints only):
        Transforms an end effector pose from 2D cartesian space into the joint space.
        Params:
            poses: nx3-matrix: End effector poses in 2D cartesian space
                  n: number of samples
                  column 1: x position of the end effector
                  column 2: y position of the end effector
                  column 3: orientation of the end effector (rad)
            link_lengths: 3x1-vector: lengths of the links (should have same unit as x and y position)
        Returns:
            joint_angles1, joint_angles2: nx3-matrices: Both IK solutions
               n: number of samples
               columns: joint angles (rad) (if no solution exists, the value will be np.NaN)
    """
    poses = np.atleast_2d(poses)
    link_lengths = np.array(link_lengths)
    assert poses.shape[1] == 3, "A Pose has to consist of three elements (x, y, alpha)"
    assert link_lengths.shape[0] == 3, "The number of link lengths has to be 3 (3-DOF manipulator)"
    n_samples = poses.shape[0]
    
    joint_angles1 = np.zeros((n_samples,3))
    joint_angles2 = np.zeros((n_samples,3))
    
    sample_times = []
    for i in range(n_samples):
        begin = timer()
        
        #rename important variables to improve readability
        l1 = link_lengths[0]
        l2 = link_lengths[1]
        l3 = link_lengths[2]
        Px = poses[i,0]
        Py = poses[i,1]
        alpha = poses[i,2]
        
        #calculate elements of type 6 equation
        Sa = math.sin(alpha)
        Ca = math.cos(alpha)
        X = Py - l3 * Sa
        Y = l3 * Ca - Px
        Z2 = -l1
        W = l2
        
        #calculate elements of the corresponding type 2 equation
        B1 = 2 * Z2 * X
        B2 = -2 * Z2 * Y
        B3 = W**2 - X**2 - Y**2 - Z2**2
        
        #solve type 2 equation
        sqroot_term = B1**2 + B2**2 - B3**2
        if sqroot_term < 0 and sqroot_term > -1e-14:
            sqroot_term = 0
        if sqroot_term < 0:
            #in this case, there is no solution (end effector out of workspace)
            joint_angles1[i,:] = np.NaN
            joint_angles2[i,:] = np.NaN
            continue
        sqroot = math.sqrt(sqroot_term)
        denom = B1**2 + B2**2
        #first solution
        S1_1 = (B1 * B3 + B2 * sqroot) / denom
        C1_1 = (B2 * B3 - B1 * sqroot) / denom
        theta1_1 = math.atan2(S1_1, C1_1)
        joint_angles1[i,0] = theta1_1
        #second solution
        S1_2 = (B1 * B3 - B2 * sqroot) / denom
        C1_2 = (B2 * B3 + B1 * sqroot) / denom
        theta1_2 = math.atan2(S1_2, C1_2)
        joint_angles2[i,0] = theta1_2
        
        #calculate elements of type 3 equation for both solution branches
        A_1 = math.cos(theta1_1) * X + math.sin(theta1_1) * Y
        B_1 = math.sin(theta1_1) * X - math.cos(theta1_1) * Y + Z2
        A_2 = math.cos(theta1_2) * X + math.sin(theta1_2) * Y
        B_2 = math.sin(theta1_2) * X - math.cos(theta1_2) * Y + Z2
        
        #solve type 3 equation
        theta2_1 = math.atan2(A_1 / l2, B_1 / l2)
        joint_angles1[i,1] = theta2_1
        theta2_2 = math.atan2(A_2 / l2, B_2 / l2)
        joint_angles2[i,1] = theta2_2
        
        #get the last joint angle via additional sin/cos constraints
        t12_1 = theta1_1 + theta2_1
        C12_1 = math.cos(t12_1)
        S12_1 = math.sin(t12_1)
        C3_1 = C12_1 * Ca + S12_1 * Sa
        S3_1 = C12_1 * Sa - S12_1 * Ca
        theta3_1 = math.atan2(S3_1, C3_1)
        joint_angles1[i,2] = theta3_1
        t12_2 = theta1_2 + theta2_2
        C12_2 = math.cos(t12_2)
        S12_2 = math.sin(t12_2)
        C3_2 = C12_2 * Ca + S12_2 * Sa
        S3_2 = C12_2 * Sa - S12_2 * Ca
        theta3_2 = math.atan2(S3_2, C3_2)
        joint_angles2[i,2] = theta3_2
        
        end = timer()
        time_sample = end - begin
        sample_times.append(time_sample)
        
    print("Sum analytical IK: "+str(np.sum(sample_times))+" seconds")
    print("Mean analytical IK: "+str(np.mean(sample_times))+" seconds")
    print("Std analytical IK: "+str(np.std(sample_times))+" seconds")
        
    return joint_angles1, joint_angles2


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


def mae_position(pos_true, pos_pred):
    """ Calculates the mean absolute position error in cartesian space. Precisely, the
        calculated error is the mean of the euclidean distances between the predicted and
        true positions.
        Params:
            pos_true: nx2-matrix: True end effector positions in 2D cartesian space
                      n: number of samples
                      column 1: true x position of the end effector
                      column 2: true y position of the end effector
            pos_pred: nx2-matrix: Predicted end effector positions in 2D cartesian space
                      n: number of samples
                      column 1: predicted x position of the end effector
                      column 2: predicted y position of the end effector
        Returns:
            mae_pos: scalar: Mean of the euclidean distances between predicted and
                             true positions (same unit as input)
    """
    mae_pos = np.mean(np.sqrt(np.sum(np.square(pos_pred - pos_true), axis=1)))
    return mae_pos


def mae_orientation_2D(orient_true, orient_pred):
    """ Calculates the mean absolute orientation error in 2D cartesian space.
        Params:
            orient_true: nx1-vector: True end effector orientations (rad)
                         n: number of samples
            orient_pred: nx1-vector: Predicted end effector orientations (rad)
                              n: number of samples
        Returns:
            mae_orient: scalar: Mean absolute orientation error (rad)
    """
    orient_diff = (orient_pred - orient_true) % (2*math.pi)
    #Normalization to [-Pi,Pi] is important here. Otherwise, the errors will be too high.
    orient_diff[orient_diff > math.pi] -= 2*math.pi
    mae_orient = np.mean(np.abs(orient_diff))
    return mae_orient


def jacobian_2D_3DOF(joint_configurations, link_lengths):
    """ Calculates the jacobian matrix of a planar 3-DOF-arm for the given joint configurations.
        Params:
            joint_configurations: nx3-matrix: angles of the three joints (rad)
                                  n: number of samples
            link_lengths: 3x1-vector: lengths of the links
        Returns:
            J: nx3x3-matrix: the jacobian matrices for the given joint angle vectors
    """
    J = np.zeros((joint_configurations.shape[0],3,3))
    s1 =   np.sin(joint_configurations[:,0])
    s12 =  np.sin(np.sum(joint_configurations[:,:2], axis=1))
    s123 = np.sin(np.sum(joint_configurations, axis=1))
    c1 =   np.cos(joint_configurations[:,0])
    c12 =  np.cos(np.sum(joint_configurations[:,:2], axis=1))
    c123 = np.cos(np.sum(joint_configurations, axis=1))
    J[:,0,0] = -s1*link_lengths[0]-s12*link_lengths[1]-s123*link_lengths[2]
    J[:,0,1] = -s12*link_lengths[1]-s123*link_lengths[2]
    J[:,0,2] = -s123*link_lengths[2]
    J[:,1,0] = c1*link_lengths[0]+c12*link_lengths[1]+c123*link_lengths[2]
    J[:,1,1] = c12*link_lengths[1]+c123*link_lengths[2]
    J[:,1,2] = c123*link_lengths[2]
    J[:,2,:] = 1
    
    return J


def min_error_2D_3DOF(joint_configurations, theta_stds, link_lengths, test=True):
    """ Calculates a comparative 2D cartesian error for a 3 DOF manipulator
        (revolute joints only) with regard to the joint accuracies (theta_stds) and joint configurations.
        Uses the jacobian of the manipulator.
        Params:
            joint_configurations: nx3-matrix: angles of the three joints (rad)
                                  n: number of samples
            theta_stds: 3x1-vector: standard deviations of the joint accuracies of the manipulator (rad)
            link_lengths: 3x1-vector: lengths of the links
            test: bool: whether to test the constructed jacobians for correctness
        Returns:
            mean_min_cartesian_errors: 2x1-vector: comparative mean of the euclidean position error
                                       (same unit as link_lengths) and the orientation error (rad)
            median_min_cartesian_errors: 2x1-vector: comparative median of the euclidean position error
                                         (same unit as link_lengths) and the orientation error (rad)
    """
    if np.any(np.isnan(joint_configurations)):
        return [np.nan, np.nan], [np.nan, np.nan]
    
    cartesian_errors = np.zeros((joint_configurations.shape[0],3))
    eucl_cartesian_errors = np.zeros((joint_configurations.shape[0],2))
    
    #construct jacobian matrix
    J = jacobian_2D_3DOF(joint_configurations, link_lengths)
        
    if test:
        #check if the first jacobian is correct
        test_jacobian_2D_3DOF(J[0], joint_configurations[0], link_lengths)
    
    #draw delta_thetas from a gaussian distribution
    np.random.seed(0)
    mean_error_sum = 0
    median_error_sum = 0
    num_draws = 20
    for k in range(num_draws):
        delta_theta = np.zeros((3,joint_configurations.shape[0]))
        delta_theta[0,:] = np.random.normal(loc=0, scale=theta_stds[0], size=(joint_configurations.shape[0]))
        delta_theta[1,:] = np.random.normal(loc=0, scale=theta_stds[1], size=(joint_configurations.shape[0]))
        delta_theta[2,:] = np.random.normal(loc=0, scale=theta_stds[2], size=(joint_configurations.shape[0]))
        
        #delta_x = J * delta_theta
        for i in range(joint_configurations.shape[0]):
            cartesian_errors[i,:] = np.dot(J[i], delta_theta[:,i])
        eucl_cartesian_errors[:,0] = np.sqrt(np.square(cartesian_errors[:,0])+np.square(cartesian_errors[:,1]))
        eucl_cartesian_errors[:,1] = np.abs(cartesian_errors[:,2])
            
        mean_min_cartesian_errors_tmp = np.mean(eucl_cartesian_errors, axis=0)
        median_min_cartesian_errors_tmp = np.median(eucl_cartesian_errors, axis=0)
        mean_error_sum += mean_min_cartesian_errors_tmp
        median_error_sum += median_min_cartesian_errors_tmp
        
    mean_min_cartesian_errors = mean_error_sum / num_draws
    median_min_cartesian_errors = median_error_sum / num_draws
    return mean_min_cartesian_errors, median_min_cartesian_errors


def test_jacobian_2D_3DOF(J, joint_configuration, link_lengths):
    """ Checks a jacobian for correctness. This function compares delta_x1 = J(theta) * delta_theta
        and delta_x2 = FK(theta) - FK(theta + delta_theta) for a very small vector delta_theta.
        If delta_x1 and delta_x2 are not close, an exception is thrown.
        Params:
            J: 3x3-matrix: the Jacobian (at joint_configuration) to test
            joint_configuration: 3x1-vector: angles of the three joints (rad)
            link_lengths: 3x1-vector: lengths of the links
    """
    delta_theta_small = np.zeros((3,1))
    std = 1e-7
    delta_theta_small[0,0] = np.random.normal(loc=0, scale=std, size=1)
    delta_theta_small[1,0] = np.random.normal(loc=0, scale=std, size=1)
    delta_theta_small[2,0] = np.random.normal(loc=0, scale=std, size=1)
    old_jc = joint_configuration
    new_jc = joint_configuration + delta_theta_small.T
    true_cart = FK_2D(old_jc, link_lengths)
    new_cart = FK_2D(new_jc, link_lengths)
    pos_error = mae_position(true_cart[:,:2], new_cart[:,:2])
    ori_error = mae_orientation_2D(true_cart[:,2], new_cart[:,2])
    cart_error_j = np.dot(J, delta_theta_small)
    pos_error_j = np.sqrt(np.square(cart_error_j[0])+np.square(cart_error_j[1]))
    ori_error_j = np.abs(cart_error_j[2])
    threshold = 1e-12
    assert np.abs(pos_error - pos_error_j[0]) < threshold, "Jacobian test failed"
    assert np.abs(ori_error - ori_error_j[0]) < threshold, "Jacobian test failed"


def joint_limit_violation_rate(predictions, limits):
    """ Calculates the joint limit violation rate.
        This is a fraction of (infeasible configurations) / (total number of configurations).
        Params:
            predictions: nxk-matrix: angles of the joints
                         n: number of samples
                         k: number of joints
            limits: kx2 array:
                    rows: joints
                    column 1: lower joint limit (same unit as predictions)
                    column 2: upper joint limit (same unit as predictions)
        Returns:
            violation_rate: scalar: rate of configurations which violate the joint limits.
    """
    bool_violations = [False]*predictions.shape[0]
    for joint_ind in range(limits.shape[0]):
        bool_violations = np.logical_or(bool_violations,
                                        predictions[:,joint_ind] < limits[joint_ind,0])
        bool_violations = np.logical_or(bool_violations,
                                        predictions[:,joint_ind] > limits[joint_ind,1])
    violation_rate = np.sum(bool_violations) / predictions.shape[0]
    return violation_rate


def plot_configurations_2D(configurations, link_lengths):
    """ Plots multiple manipulator configurations (revolute joints only) into one 2D figure.
        Params:
            configurations: list of nxk-matrices: angles of the joints (rad)
                            n: number of samples
                            k: number of joints
            link_lengths: kx1-vector: lengths of the links
                          k: number of joints (=number of links)
    """
    configurations = np.atleast_2d(configurations)
    link_lengths = np.array(link_lengths)
    n_DOFS = []
    for theta in configurations:
        n_DOFS.append(theta.shape[0])
    assert np.all(np.array(n_DOFS) == link_lengths.shape[0]), "Numbers of joint angles and link lengths have to be equal"
    n_DOFS = link_lengths.shape[0]
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.figure()
    for color_ind, theta in enumerate(configurations):
        curr_color = color_list[color_ind % len(color_list)]
        x1 = 0
        y1 = 0
        for j in range(n_DOFS):
            x2 = x1 + np.cos(np.sum(theta[:j+1]))*link_lengths[j]
            y2 = y1 + np.sin(np.sum(theta[:j+1]))*link_lengths[j]            
            plt.plot([x1, x2], [y1, y2], color=curr_color, label="_")
            x1 = x2
            y1 = y2
            
        #draw the end effector as a circle
        plt.scatter(x2, y2, color=curr_color, label=str(color_ind))
        
        #draw an arrow to indicate end effector orientation
        dx = np.cos(np.sum(theta))*0.05*np.sum(link_lengths)
        dy = np.sin(np.sum(theta))*0.05*np.sum(link_lengths)
        plt.arrow(x2, y2, dx, dy, head_width=0.02*np.sum(link_lengths), color=curr_color, label="_")
        
    return plt.gcf(), plt.gca()


def pos_error_distribution(pos_true, pos_pred, plot=False):
    """ Calculates the position errors for the given samples and sorts them.
        Params:
            pos_true: nx2-matrix: true end effector positions
            pos_pred: nx2-matrix: predicted end effector positions
            plot: bool: whether to plot the sorted errors
        Returns:
            x: nx1-vector: sorted position errors
    """
    errs = np.sqrt(np.sum(np.square(pos_pred - pos_true), axis=1))
    x = np.sort(errs)
    if plot:
        y = (1+np.arange(x.shape[0])) / x.shape[0]
        plt.figure()
        plt.plot(x,y)
        plt.xlabel("Position error (m)")
        plt.ylabel("Percentile")
    return x
    

def ori_error_distribution(orient_true, orient_pred, plot=False):
    """ Calculates the orientation errors for the given samples and sorts them.
        Params:
            orient_true: nx1-vector: true end effector orientations
            orient_pred: nx1-vector: predicted end effector orientations
            plot: bool: whether to plot the sorted errors
        Returns:
            x: nx1-vector: sorted orientation errors
    """
    orient_diff = (orient_pred - orient_true) % (2*math.pi)
    #Normalization to [-Pi,Pi] is important here. Otherwise, the errors will be too high.
    orient_diff[orient_diff > math.pi] -= 2*math.pi
    x = np.sort(np.abs(orient_diff))
    if plot:
        y = (1+np.arange(x.shape[0])) / x.shape[0]
        plt.figure()
        plt.plot(x,y)
        plt.xlabel("Orientation error (rad)")
        plt.ylabel("Percentile")
    return x
    
  
def get_solve_rate(cartesian_true, cartesian_pred, pos_err_threshold, ori_err_threshold):
    """ Calculates the solve rate for a given position and orientation threshold. A prediction is only a valid solution
        if it leads to a smaller error with regards to both thresholds.
        Params:
            cartesian_true: nx3-matrix: true end effector pose (2D)
            cartesian_pred: nx3-matrix: predicted end effector pose (2D)
        Returns:
            solve_rate: number of valid solutions / number of total predictions
    """
    pos_errs = np.sqrt(np.sum(np.square(cartesian_pred[:,:2] - cartesian_true[:,:2]), axis=1))
    orient_diff = (cartesian_pred[:,2] - cartesian_true[:,2]) % (2*math.pi)
    #Normalization to [-Pi,Pi] is important here. Otherwise, the errors will be too high.
    orient_diff[orient_diff > math.pi] -= 2*math.pi
    ori_errs = np.abs(orient_diff)
    number_of_solutions = np.sum(np.logical_and(pos_errs < pos_err_threshold, ori_errs < ori_err_threshold))
    solve_rate = number_of_solutions / pos_errs.shape[0]
    return solve_rate


def cartesian_error_sincos(y_true, y_pred, link_lengths, w, joint_limits, joint_penalty, penalty_factor):
    """ This loss function transforms the predicted joint angles via the FK model
        into cartesian space where the error is calculated.
        Params:
            y_true: Tensor: The desired pose in cartesian space (orientation in sin/cos representation)
            y_pred: Tensor: The predicted joint angles in joint space (in sin/cos representation)
            link_lengths: array-like: lengths of the links
            w: weight of the position error; weight of orientation error is (1-w)
            joint_limits: kx2 array:
                          rows: joints
                          column 1: lower joint limit (rad)
                          column 2: upper joint limit (rad)
            joint_penalty: Bool: add additional penalty term to loss in case of violations of joint limits
            penalty_factor: scalar: defines how strong the joint limit violation will be weighted
        Returns:
            loss: Loss in cartesian space. This is a weighted sum of the position loss
                  (mean euclidean distance) and the orientation loss (mean absolute orientation error).
                  If enabled, an extra amount of loss is added in case of joint limit violations.
    """
    n_DOFS = np.array(link_lengths).shape[0]
    y_pred_scaled = y_pred
    
    j = []
    for i in range(n_DOFS):
        j.append(tf.atan2(y_pred_scaled[:,i], y_pred_scaled[:,i+n_DOFS]))
    joints = tf.stack(j, axis=-1)
            
    #1. position loss (mean euclidean distance)
    x_summands = []
    y_summands = []
    for i in range(n_DOFS):
        x_summands.append(tf.cos(tf.reduce_sum(joints[:,:i+1], axis=1)) * link_lengths[i])
        y_summands.append(tf.sin(tf.reduce_sum(joints[:,:i+1], axis=1)) * link_lengths[i])
        
    x_pred = tf.reduce_sum(tf.stack(x_summands, axis=-1), axis=1)
    y_pred = tf.reduce_sum(tf.stack(y_summands, axis=-1), axis=1)
    
    pos_true = y_true[:,:2]
    pos_pred = tf.stack([x_pred,y_pred], axis=-1)
    loss_pos = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(pos_pred - pos_true), axis=1)))
    #for mean squared error use instead:
    #loss_pos = tf.reduce_mean(tf.reduce_sum(tf.square(pos_pred - pos_true), axis=1))
    
    #2. orientation loss (mean absolute orientation error)
    alpha_pred = tf.reduce_sum(joints, axis=1)
    alpha_true = tf.atan2(y_true[:,2], y_true[:,3])
    diff_orient_normal = normalize_tensor_angle(alpha_pred - alpha_true)
    loss_orient = tf.reduce_mean(tf.abs(diff_orient_normal))
    #for mean squared error use instead:
    #loss_orient = tf.reduce_mean(tf.square(diff_orient_normal))
    
    #3. weighted sum with hyperparameter w
    loss = w * loss_pos + (1-w) * loss_orient
    
    #4. if enabled, add loss term in case of joint limit violations
    if joint_penalty:
        penalties_list = []
        for i in range(n_DOFS):
            center = tf.convert_to_tensor(joint_limits[i,1] + joint_limits[i,0], dtype=tf.float32) / 2  #center of feasible space
            diff_to_center = normalize_tensor_angle(joints[:,i] - center)
            
            #if diff_to_center is positive: actual angle is closer to upper joint limit
            #if diff_to_center is negative: actual angle is closer to lower joint limit
            greater_zero = tf.greater(diff_to_center, 0)
            diff_to_lower = normalize_tensor_angle(joint_limits[i,0] - joints[:,i])
            diff_to_upper = normalize_tensor_angle(joints[:,i] - joint_limits[i,1])
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
        
        loss_with_penalty = loss + penalty_factor * (penalty_sum / tf.cast(tf.shape(joints)[0], tf.float32))
        return loss_with_penalty

    else:       
        return loss


def normalize_tensor_angle(angle):
    """ Normalizes angles stored in a tensor.
        Params:
            angle: Tensor: The angles to normalize (rad)
        Returns:
            angle_normal: Tensor: The normalized angles in the range [-pi, pi] (rad)
    """
    angle_mod = tf.mod(angle, 2*math.pi)
    angle_minus_twopi = angle_mod - 2*math.pi
    greater_pi = tf.greater(angle_mod, math.pi)
    angle_normal = tf.where(greater_pi, angle_minus_twopi, angle_mod)
    return angle_normal


def cartesian_loss_sincos(link_lengths, w, joint_limits, joint_penalty, penalty_factor):
    """ Wrapper function for the cartesian error loss function used for
        distal teaching.
    """
    def cart_loss(y_true, y_pred):
        return cartesian_error_sincos(y_true, y_pred, link_lengths, w, joint_limits, joint_penalty, penalty_factor)
    return cart_loss
        