import math
import numpy as np
import tensorflow as tf
from pytransform3d.urdf import UrdfTransformManager
from helpers import quaternion_from_matrix
from matplotlib import pyplot as plt

class Atlas():
    """ This class represents a 15 DOF (revolute joints only) serial chain of the humanoid Atlas.
        The investigated chain is the following:
        ['l_foot', 'l_talus', 'l_lleg', 'l_uleg', 'l_lglut', 'l_uglut', 'pelvis', 'ltorso', 'mtorso',\
         'utorso', 'l_clav', 'l_scap', 'l_uarm', 'l_larm', 'l_farm', 'l_hand', 'tcp']
    """
    
    def __init__(self, urdf_path, theta_stds=None):
        """ Constructor:
            Params:
                urdf_path: string: path to the URDF file containing the model of the manipulator
                           including all relevant parameters
                theta_stds: 15x1-vector: standard deviations of the joint accuracies of the manipulator (rad)
        """
        with open(urdf_path) as f:
            urdf_string = f.read()
        self.tm = UrdfTransformManager()
        self.tm.load_urdf(urdf_string)
        self.n_DOFS = 15
        self.has_jacobian = False
        self.has_IK = False
        self.has_singularity_condition = False
        
        #link names of the investigated subchain of Atlas from base to end effector (15-DOF)
        self.sorted_link_names = ['l_foot', 'l_talus', 'l_lleg', 'l_uleg', 'l_lglut', 'l_uglut', 'pelvis', 'ltorso', 'mtorso',\
                                  'utorso', 'l_clav', 'l_scap', 'l_uarm', 'l_larm', 'l_farm', 'l_hand', 'tcp']
        
        self.joint_info = self._extract_relevant_joints()
         
        #joint names of the investigated subchain of Atlas from base to end effector (15-DOF)
        self.sorted_joint_names = self._get_sorted_joint_names()
        
        if theta_stds is None:
            #default is an uncertainty of a standard deviation of 1 degree per joint
            self.theta_stds = np.zeros(self.n_DOFS) + (1*math.pi/180)
        else:
            self.theta_stds = theta_stds
            
        self.joint_transformations_notheta = self._construct_transformations_notheta()
        self.summands2, self.summands3 = self._prepare_rodrigues_summands()
        
        #trajectory that is tested in one of the experiments
        #numbers are: [start_x, end_x, start_y, end_y, start_z, end_z]
        self.trajectory = [-0.5, -0.1, -0.3, 0.3, 1.3, 1.8]
        
        #arbitrarily chosen subspace of the maximum workspace (for no joint limits) where we train and test
        #it is secured via numerical tests that this set really lies in the workspace
        self.subspace = np.array([[0.7, 1.2],
                                  [-0.4, 0.1],
                                  [1, 1.5],
                                  [0.6, 0.6],
                                  [0.1, 0.1],
                                  [0.1, 0.1],
                                  [0, 0.5]])
   
    
    def _extract_relevant_joints(self):
        """ Removes the joints not belonging to the investigated chain of Atlas.
        """
        joint_info = self.tm._joints.copy()
        for joint_name in list(self.tm._joints):
            if self.tm._joints[joint_name][0] not in self.sorted_link_names or self.tm._joints[joint_name][1] not in self.sorted_link_names:
                joint_info.pop(joint_name)
        return joint_info
            
            
    def _get_sorted_joint_names(self):
        """ Collects the joint names in a sorted list where the order corresponds to the the chain from the
            robot base to the end effector.
        """
        sorted_joints = []
        for i in range(len(self.sorted_link_names)-1):
            for joint in self.joint_info:
                if self.sorted_link_names[i] in self.joint_info[joint][:2] and self.sorted_link_names[i+1] in self.joint_info[joint][:2]:
                    sorted_joints.append(joint)
        return sorted_joints
    
    
    def _construct_transformations_notheta(self):
        """ Prepares the transformation matrices for this manipulator between two adjacent joints.
            They are valid only for theta_i = 0 for all i, i.e. the default configuration. Later, these
            matrices are re-used in the FK calculation.
            Returns:
                transformations_notheta: 7x4x4-matrix: contains the transformation matrices for adjacent joints
                                                       in the case of theta_i = 0 for all i
        """    
        transformations_notheta = np.zeros((self.n_DOFS+1, 4, 4))
        for i, joint in enumerate(self.sorted_joint_names):
            #take the child to parent transformation matrices for the default configuration
            transformations_notheta[i] = self.joint_info[joint][2]
        #static tcp transformation
        transformations_notheta[-1] = self.tm.get_transform(self.sorted_link_names[-1], self.sorted_link_names[-2])

        return transformations_notheta
    
    
    def _prepare_rodrigues_summands(self):
        """ Prepares the second summand of the decomposed Rodrigues formula which is used in the FK calculation.
            This matrix does not depend on theta, therefore it is pre-calculated to speed up the FK calculation at runtime.
            Returns: the nx4x4 matrices for the second and third summand of the decomposed Rodrigues formula
        """
        #source of the decomposed Rodrigues formula: https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/
        
        summand2 = np.zeros((self.n_DOFS, 4, 4))
        summand3 = np.zeros((self.n_DOFS, 4, 4))
        for i, joint in enumerate(self.sorted_joint_names):
            #get rotation axis of the joint
            x, y, z = self.joint_info[joint][3]
            summand2[i,:3,:3] = np.array([[x*x, x*y, x*z],
                                          [x*y, y*y, y*z],
                                          [x*z, y*z, z*z]])
            summand3[i,:3,:3] = np.array([[0, -z, y],
                                          [z, 0, -x],
                                          [-y, x, 0]])
        return summand2, summand3
    
    
    def _test_jacobian(self, J, theta_vec):
        """ Checks a jacobian for correctness. This function compares delta_x1 = J(theta) * delta_theta
            and delta_x2 = FK(theta) - FK(theta + delta_theta) for a very small vector delta_theta.
            If delta_x1 and delta_x2 are not close, an exception is thrown.
            Params:
                J: 6x15-matrix: The manipulator jacobian at theta_vec
                theta_vec: 1x15-vector: The joint angles for which the Jacobian is valid
        """
        theta_vec = np.atleast_2d(theta_vec)
        delta_theta_small = np.zeros((self.n_DOFS,1))
        std = 1e-4
        delta_theta_small[:,0] = np.random.normal(loc=0, scale=std, size=self.n_DOFS)
        old_jc = theta_vec
        new_jc = theta_vec + delta_theta_small.T
        true_T = self.FK(old_jc)[0]
        new_T = self.FK(new_jc)[0]
        true_quat = quaternion_from_matrix(true_T[:3,:3])
        new_quat = quaternion_from_matrix(new_T[:3,:3])
        pos_error = np.sqrt(np.sum(np.square(new_T[:3,3] - true_T[:3,3])))
        ori_error = 2 * np.arccos(np.abs(np.inner(new_quat, true_quat)))
        cart_error_j = np.dot(J, delta_theta_small)
        pos_error_j = np.sqrt(np.sum(np.square(cart_error_j[:3])))
        ori_error_j = np.sqrt(np.sum(np.square(cart_error_j[3:])))
        threshold = 1e-7
        assert np.abs(pos_error - pos_error_j) < threshold, "Jacobian test failed"
        assert np.abs(ori_error - ori_error_j) < threshold, "Jacobian test failed"
    
    
    def FK(self, thetas):
        """ Wrapper for the Forward Kinematic Model which transforms joint configurations
            into cartesian space.
            Params:
                thetas: nx15-matrix or -tensor: angles of the joints (rad)
                        rows: samples
                        columns: joints
            Returns:
                Ts: nx4x4-matrix: homogenous transformation matrices
                    n: number of samples
                    X[i,:,:]: homogenous transformation matrix of the ith sample
                              representing the end effector pose
        """
        sess = tf.Session()
        with sess.as_default():
            thetas_T = tf.convert_to_tensor(thetas)
            Ts = self.FK_tensor(thetas_T).eval()
        return Ts
    
    
    def FK_tensor(self, thetas):
        """ Forward Kinematic Model:
            Transforms joint configurations into 3D cartesian space.
            Params:
                thetas: nx15-tensor: angles of the joints (rad)
                        rows: samples
                        columns: joints
            Returns:
                _0_T_tcp: nx4x4-tensor: homogenous transformation matrices
                          n: number of samples
                          X[i,:,:]: homogenous transformation matrix of the ith sample
                                    representing the end effector pose
        """
        #make sure the datatypes are all float64
        thetas = tf.cast(thetas, dtype=tf.float64)
        
        #convert preliminary transformation matrices to tensors
        m1 = [tf.shape(thetas)[0],1,1,1]
        m2 = [tf.shape(thetas)[0],1,1]
        trans_notheta = tf.tile(tf.expand_dims(tf.convert_to_tensor(self.joint_transformations_notheta[:15], dtype=tf.float64), 0), m1) #nx16x4x4
        trans_tcp = tf.tile(tf.expand_dims(tf.convert_to_tensor(self.joint_transformations_notheta[15], dtype=tf.float64) , 0), m2)     #nx4x4
        
        #use the decomposed Rodrigues formula (source: https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/)
        identity = tf.tile(tf.expand_dims(tf.eye(4, batch_shape=[15], dtype=tf.float64), 0), m1) #nx15x4x4
        summands2 = tf.expand_dims(tf.convert_to_tensor(self.summands2, dtype=tf.float64), 0)    #1x15x4x4
        summands3 = tf.expand_dims(tf.convert_to_tensor(self.summands3, dtype=tf.float64), 0)    #1x15x4x4
        
        Sth = tf.reshape(tf.sin(thetas), [-1,15,1,1]) #nx15x1x1
        Cth = tf.reshape(tf.cos(thetas), [-1,15,1,1]) #nx15x1x1
        tth = 1 - Cth                                 #nx15x1x1
        
        scaled_identity = tf.multiply(Cth, identity[:,:,:3,:3]) #nx15x3x3
        tmp = tf.concat([scaled_identity, identity[:,:,:3,3:]], axis=-1) #nx15x3x4
        scaled_homogenous_summands1 = tf.concat([tmp, identity[:,:,3:,:]], axis=2) #nx15x4x4
        
        trans_theta = scaled_homogenous_summands1 + tf.multiply(tth, summands2) + tf.multiply(Sth, summands3) #nx16x4x4
        
        _i_T_ip1 = tf.matmul(trans_notheta, trans_theta)                                #nx15x4x4
        
        #matrix multiplication of all transformation matrices
        _2_T_0 = tf.matmul(_i_T_ip1[:,1,:,:], _i_T_ip1[:,0,:,:])                        #nx4x4
        _4_T_2 = tf.matmul(_i_T_ip1[:,3,:,:], _i_T_ip1[:,2,:,:])                        #nx4x4
        _6_T_4 = tf.matmul(_i_T_ip1[:,5,:,:], _i_T_ip1[:,4,:,:])                        #nx4x4
        _6_T_0 = tf.matmul(_6_T_4, tf.matmul(_4_T_2, _2_T_0))
        #this matrix needs to be inversed because the investigated chain spans two branches of the structural tree of the URDF.
        #the root node (index 6) is the pelvis
        #from index 0 to 6 we go upwards from a leaf towards the root node
        #from index 6 to 16 we go downwards from the root towards another leaf
        #due to this, the order of parent and child changes which makes the inversion necessary at this point
        #homogenous transformation matrices can be inverted faster by using a transposition:
        R_t = tf.matrix_transpose(_6_T_0[:,:3,:3])   #nx3x3
        t = _6_T_0[:,:3,3:]                          #nx3x1
        mRt = - tf.matmul(R_t, t)                    #nx3x1
        row4 = identity[:,0,3:,:]                    #nx1x4
        row123 = tf.concat([R_t, mRt], axis=-1)      #nx3x4
        _0_T_6 = tf.concat([row123, row4], axis=1)   #nx4x4
        
        
        _6_T_8 = tf.matmul(_i_T_ip1[:,6,:,:], _i_T_ip1[:,7,:,:])                        #nx4x4
        _8_T_10 = tf.matmul(_i_T_ip1[:,8,:,:], _i_T_ip1[:,9,:,:])                       #nx4x4
        _10_T_12 = tf.matmul(_i_T_ip1[:,10,:,:], _i_T_ip1[:,11,:,:])                    #nx4x4
        _12_T_14 = tf.matmul(_i_T_ip1[:,12,:,:], _i_T_ip1[:,13,:,:])                    #nx4x4
        _14_T_tcp = tf.matmul(_i_T_ip1[:,14,:,:], trans_tcp)                            #nx4x4
        
        _6_T_10 = tf.matmul(_6_T_8, _8_T_10)
        _10_T_14 = tf.matmul(_10_T_12, _12_T_14)
        
        _0_T_10 = tf.matmul(_0_T_6, _6_T_10)
        _10_T_tcp = tf.matmul(_10_T_14, _14_T_tcp)
        
        _0_T_tcp = tf.matmul(_0_T_10, _10_T_tcp)
        
        return _0_T_tcp
    
    
    def plot_configurations(self, joint_configurations):
        """ Plots manipulator configurations in 3D cartesian space.
            Params:
                joint_configurations: nx15-matrix: angles of the joints (rad)
                                      n: number of samples
            Returns:
                the figure and axis references
        """
        for i in range(joint_configurations.shape[0]):
            joint_names = self.sorted_joint_names
            joint_angles = joint_configurations[i,:]
            for name, angle in zip(joint_names, joint_angles):
                self.tm.set_joint(name, angle)
            if i == 0:
                ax = self.tm.plot_frames_in("l_foot", s=0.05, show_name=False, whitelist=self.sorted_link_names)
            else:
                ax = self.tm.plot_frames_in("l_foot", s=0.05, ax=ax, show_name=False, whitelist=self.sorted_link_names)
            ax = self.tm.plot_connections_in("l_foot", ax=ax, whitelist=self.sorted_link_names)
        
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        fig = plt.gcf()
        return fig, ax
   