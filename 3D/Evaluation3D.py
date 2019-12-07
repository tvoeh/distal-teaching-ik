import numpy as np
from matplotlib import pyplot as plt
from helpers import quaternion_from_matrix

class Evaluation3D():
    """ This class provides several evaluation metrics in 3D cartesian space and
        in the joint space.
    """
    
    def __init__(self, robot, pred_cartesian, pred_joints, true_cartesian, true_joints):
        """ Constructor:
            Params:
                robot: object of the robot (e.g. class Compi) which contains the relevant robot parameters
                pred_cartesian: nx4x4- or nx7-matrix: predicted end effector poses in cartesian space,
                                either homogenous transformation matrices or positions+quaternions.
                                In the latter case, the first three columns have to represent the position.
                                n: number of samples
                pred_joints:    nxk-matrix: predictions in joint space (rad)
                                rows: samples
                                columns: joints
                true_cartesian: nx4x4- or nx7-matrix: true end effector poses in cartesian space,
                                either homogenous transformation matrices or positions+quaternions.
                                In the latter case, the first three columns have to represent the position.
                                n: number of samples
                true_joints:    nxk-matrix: true joint angles (rad)
                                rows: samples
                                columns: joints
        """
        self.robot = robot
        self.pred_joints = pred_joints
        self.pred_cartesian = pred_cartesian
        self.true_joints = true_joints
        self.true_cartesian = true_cartesian
        
        if len(pred_cartesian.shape) == 3:
            #transformation matrix was given
            #convert it to position + quaternion
            pos_quat = np.zeros((pred_cartesian.shape[0], 7))
            pos_quat[:,:3] = pred_cartesian[:,:3,3]
            pos_quat[:,3:] = quaternion_from_matrix(pred_cartesian[:,:3,:3])
            self.pred_cartesian = pos_quat
            
        if len(true_cartesian.shape) == 3:
            #transformation matrix was given
            #convert it to position + quaternion
            pos_quat = np.zeros((true_cartesian.shape[0], 7))
            pos_quat[:,:3] = true_cartesian[:,:3,3]
            pos_quat[:,3:] = quaternion_from_matrix(true_cartesian[:,:3,:3])
            self.true_cartesian = pos_quat
    
    
    def mae_position(self):
        """ Calculates the mean absolute position error in cartesian space. Precisely, the
            calculated error is the mean of the euclidean distances between the predicted and
            true positions.
            Returns:
                mae_pos: scalar: Mean of the euclidean distances between predicted and
                                 true positions (same unit as first three elements of class
                                 attributes pred_cartesian and true_cartesian)
        """
        mae_pos = np.mean(self.get_pos_errors())
        return mae_pos
    
    
    def mae_orientation(self):
        """ Calculates the mean absolute orientation error in 3D cartesian space. Precisely, it is the
            mean of the minimum (absolute) rotation angle required to align the predicted orientation with the
            true orientation.
            Returns:
                mae_orient: scalar: Mean absolute orientation error (rad)
        """
        
        mae_orient = np.mean(self.get_ori_errors())
        return mae_orient
    

    def get_pos_errors(self):
        """ Calculates the position errors between predicted and true poses.
            Returns:
                nx1-vector: position errors
        """
        return self._get_pos_differences(self.pred_cartesian[:,:3], self.true_cartesian[:,:3])
    
    
    def _get_pos_differences(self, pos_1, pos_2):
        """ Calculates the euclidean position differences between two poses each.
            Params:
                posquat_1: nx3-matrix: positions of pose 1
                posquat_2: nx3-matrix: positions of pose 2
            Returns:
                errs: nx1-vector: position differences
        """
        errs = np.sqrt(np.sum(np.square(pos_1 - pos_2), axis=1))
        return errs
    

    def get_ori_errors(self):
        """ Calculates the orientation errors between predicted and true poses.
            Returns:
                nx1-vector: orientation errors
        """
        return self._get_ori_differences(self.pred_cartesian[:,3:], self.true_cartesian[:,3:])
    
    
    def _get_ori_differences(self, quat_1, quat_2):
        """ Calculates the orientation differences between two poses each.
            Params:
                posquat_1: nx4-matrix: quaternions of pose 1
                posquat_2: nx4-matrix: quaternions of pose 2
            Returns:
                errs: nx1-vector: orientation differences
        """
        inner_products = np.zeros(quat_1.shape[0])
        for i in range(inner_products.shape[0]):
            inner_products[i] = np.inner(quat_1[i], quat_2[i])
            
        #make sure that rounding errors dont lead to nan values
        #1 is the maximum possible value, so clip it
        inner_products[inner_products > 1] = 1
        
        orient_diff = 2 * np.arccos(np.abs(inner_products))
        errs = np.abs(orient_diff)
        return errs
    
    
    def get_solve_rate(self, pos_err_threshold, ori_err_threshold):
        """ Calculates the solve rate for a given position and orientation threshold. A prediction is only a valid solution
            if it leads to a smaller error with regards to both thresholds.
            Returns:
                solve_rate: number of valid solutions / number of total predictions
        """
        pos_errs = self.get_pos_errors()
        ori_errs = self.get_ori_errors()
        number_of_solutions = np.sum(np.logical_and(pos_errs < pos_err_threshold, ori_errs < ori_err_threshold))
        solve_rate = number_of_solutions / pos_errs.shape[0]
        return solve_rate
    
        
    def joint_limit_violation_rate(self):
        """ Calculates the joint limit violation rate.
            This is a fraction of (infeasible configurations) / (total number of configurations).
            Returns:
                violation_rate: scalar: rate of configurations which violate the joint limits.
        """
        bool_violations = [False]*self.pred_joints.shape[0]
        for i, joint in enumerate(self.robot.sorted_joint_names):
            bool_violations = np.logical_or(bool_violations,
                                            self.pred_joints[:,i] < self.robot.joint_info[joint][4][0])
            bool_violations = np.logical_or(bool_violations,
                                            self.pred_joints[:,i] > self.robot.joint_info[joint][4][1])
        violation_rate = np.sum(bool_violations) / self.pred_joints.shape[0]
        return violation_rate
    
    
    def min_cart_error(self):
        """ Calculates the comparative 3D cartesian error resulting from the given standard deviation in joint space.
            Returns:
                mean_min_cartesian_errors: 2x1-vector: comparative mean of the euclidean position error
                                           (same unit as robot link lengths) and the orientation error (rad)
                median_min_cartesian_errors: 2x1-vector: comparative median of the euclidean position error
                                             (same unit as link_lengths) and the orientation error (rad)
        """
        
        if np.any(np.isnan(self.true_joints)):
            return [np.nan, np.nan], [np.nan, np.nan]
        
        cartesian_errors = np.zeros((self.true_joints.shape[0],6))
        eucl_cartesian_errors = np.zeros((self.true_joints.shape[0],2))
                
        #draw delta_thetas from a gaussian distribution
        np.random.seed(0)
        mean_error_sum = 0
        median_error_sum = 0
        num_draws = 1
        
        if self.robot.has_jacobian:
            #construct jacobians for given manipulator and joint configurations
            J = self.robot.jacobian(self.true_joints, test=False)
        
        for k in range(num_draws):
            delta_theta = np.zeros((self.true_joints.shape[1], self.true_joints.shape[0]))
            for i in range(self.true_joints.shape[1]):
                delta_theta[i,:] = np.random.normal(loc=0, scale=self.robot.theta_stds[i], size=(self.true_joints.shape[0]))
            
            if self.robot.has_jacobian:
                #delta_x = J * delta_theta
                for i in range(self.true_joints.shape[0]):
                    cartesian_errors[i,:] = np.dot(J[i], delta_theta[:,i])
                    
                eucl_cartesian_errors[:,0] = np.sqrt(np.sum(np.square(cartesian_errors[:,:3]), axis=1))
                
                #jacobian multiplication leads to an axis*angle representation
                #hence, calculate the minimum rotation angle via the magnitude
                eucl_cartesian_errors[:,1] = np.sqrt(np.sum(np.square(cartesian_errors[:,3:]), axis=1))
                
            else:
                #delta_x = FK(theta+delta_theta) - FK(theta)
                SE3_deltatheta = self.robot.FK(self.true_joints + delta_theta.T)
                pos_quat_deltatheta = np.zeros((SE3_deltatheta.shape[0], 7))
                pos_quat_deltatheta[:,:3] = SE3_deltatheta[:,:3,3]
                pos_quat_deltatheta[:,3:] = quaternion_from_matrix(SE3_deltatheta[:,:3,:3])
                
                eucl_cartesian_errors[:,0] = self._get_pos_differences(pos_quat_deltatheta[:,:3], self.true_cartesian[:,:3])
                eucl_cartesian_errors[:,1] = self._get_ori_differences(pos_quat_deltatheta[:,3:], self.true_cartesian[:,3:])

            
            mean_min_cartesian_errors_tmp = np.mean(eucl_cartesian_errors, axis=0)
            median_min_cartesian_errors_tmp = np.median(eucl_cartesian_errors, axis=0)
            
            mean_error_sum += mean_min_cartesian_errors_tmp
            median_error_sum += median_min_cartesian_errors_tmp
            
        mean_min_cartesian_errors = mean_error_sum / num_draws
        median_min_cartesian_errors = median_error_sum / num_draws
        return mean_min_cartesian_errors, median_min_cartesian_errors
    
    
    def plot_joint_trajectory(self):
        """ Plots the predicted joint angles as a trajectory. This is useful for the investigation of the continuity in the
            choice of solutions.
        """
        plt.figure(24)
        for i in range(self.pred_joints.shape[1]):
            plt.scatter(np.arange(self.pred_joints.shape[0]), self.pred_joints[:,i], s=12, label=r"$\theta_{"+str(i+1)+"}$")
        plt.xlabel("Query pose index k", fontsize="large")
        plt.ylabel(r"Joint angle $\theta_i$ [rad]", fontsize="large")
        plt.legend(loc="upper right", fontsize="small")
        plt.grid(True)
        plt.gcf().set_size_inches(4, 4)