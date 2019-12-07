import math
import numpy as np
import tensorflow as tf
from pytransform3d.urdf import UrdfTransformManager
from helpers import quaternion_from_matrix
from matplotlib import pyplot as plt
from timeit import default_timer as timer

class Compi():
    """ This class represents a 6 DOF RRR-RRR manipulator with a spherical wrist, e.g. Compi of the DFKI.
        Link lengths and joint limits can be varied via the input URDF-file but the joint rotation axes and
        the coordinate system relations (alpha) are fixed because analytical IK and Jacobian depend on it
        and are hardcoded. This is due to the fact that the whole derivation process, especially of the analytical IK,
        highly depends on those parameters and is difficult to generalize.
    """
    
    def __init__(self, urdf_path, theta_stds=None):
        """ Constructor:
            Params:
                urdf_path: path to the URDF file containing the model of the manipulator
                           including all relevant parameters
                theta_stds: 6x1-vector: standard deviations of the joint accuracies of the manipulator (rad)
        """
        with open(urdf_path) as f:
            urdf_string = f.read()
        self.tm = UrdfTransformManager()
        self.tm.load_urdf(urdf_string)
        self.n_DOFS = 6
        self.has_jacobian = True
        self.has_IK = True
        self.has_singularity_condition = True
        
        #sorted link names of the serial chain
        self.sorted_link_names = ['base_link', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'tcp']
        
        #sorted joint names of the serial chain
        self.sorted_joint_names = self._get_sorted_joint_names()
        
        self.joint_info = self.tm._joints.copy()
        
        if theta_stds is None:
            #default is an uncertainty of a standard deviation of 1 degree per joint
            self.theta_stds = np.zeros(self.n_DOFS) + (1*math.pi/180)
        else:
            self.theta_stds = theta_stds
        self.joint_transformations_notheta = self._construct_transformations_notheta()
        
        #trajectory that is tested in one of the experiments
        #numbers are: [start_x, end_x, start_y, end_y, start_z, end_z]
        self.trajectory = [-0.5, -0.1, -0.3, 0.3, 0.4, 0.4]
        
        #arbitrarily chosen subspace of the maximum workspace (for no joint limits) where we train and test
        #it is secured via numerical tests that this set really lies in the workspace
        self.subspace = np.array([[0, 0.5],
                                  [0, 0.3],
                                  [0.3, 0.6],
                                  [0.8, 0.8],
                                  [0.1, 0.1],
                                  [0.1, 0.1],
                                  [0, 0.3]])
        
        
    def _get_sorted_joint_names(self):
        """ Collects the joint names in a sorted list where the order corresponds to the the chain from the
            robot base to the end effector.
        """
        sorted_joints = []
        for i in range(len(self.sorted_link_names)-1):
            for joint in self.tm._joints:
                if self.sorted_link_names[i] in self.tm._joints[joint][:2] and self.sorted_link_names[i+1] in self.tm._joints[joint][:2]:
                    sorted_joints.append(joint)
        return sorted_joints
        
        
    def _construct_transformations_notheta(self):
        """ Prepares the transformation matrices for this manipulator between two adjacent joints.
            The transformation matrices are pre-filled with all constant elements, i.e. link lengths,
            rotation axes, angles alpha between two adjacent coordinate systems. The terms containing
            the angle configuration theta are left out and need to be multiplied at a later stage.
            Hence, this matrices are no valid homogenous transformation matrices in this form. However,
            they enable a fast computation of the forward kinematics because the constant parts don't need
            to be calculated over and over again.
            Returns:
                transformations_notheta: 7x4x4-matrix: contains the pre-filled matrices that can be turned
                                          into homogenous transformation matrices easily
        """
        #the construction of homogenous transformation matrices from robot parameters refers to
        #"Modeling and Control of Manipulators, Part I: Geometric and Kinematic Models" by W. Khalil, page 40
        
        #get link lengths
        l1 = self.tm.get_transform('link2','link1')[2,3]
        l2 = self.tm.get_transform('link3','link2')[0,3]
        l3 = self.tm.get_transform('link4','link3')[1,3]
        l4 = self.tm.get_transform('tcp','link6')[2,3]
        
        transformations_notheta = np.zeros((7,4,4))
        transformations_notheta[:,0,:2] = 1
        transformations_notheta[:,3,3] = 1
        
        #fill in the last columns containing the link lengths parameters
        transformations_notheta[1,2,3] = l1
        transformations_notheta[2,0,3] = l2
        transformations_notheta[3,1,3] = l3
        transformations_notheta[6,2,3] = l4
        
        #fill in the alpha-factors in the second and third columns
        alphas = [0, math.pi/2, 0, -math.pi/2, math.pi/2, -math.pi/2, 0]
        for i, alpha in enumerate(alphas):
            Ca = math.cos(alpha)
            Sa = math.sin(alpha)
            transformations_notheta[i,1,:3] = [Ca, Ca, -Sa]
            transformations_notheta[i,2,:3] = [Sa, Sa, Ca]
            
        #the last transformation (end effector) is a static one
        transformations_notheta[6,:3,:3] = np.eye(3)
            
        return transformations_notheta
    
    
    def turn_configurations_singular(self, thetas):
        #COMPI: elbow singularity: theta3=-pi/2 or theta3=pi/2
        thetas[:,2] = -math.pi/2
        return thetas


    def jacobian(self, thetas, test=False):
        """ Constructs jacobian matrices of the manipulator for given joint configurations.
            Params:
                thetas: nx6-matrix: angles of the joints (rad)
                        rows: samples
                        columns: joints
                test: bool: whether to test the constructed jacobians for correctness
            Returns:
                J: nx6x6-matrix: jacobian matrices
                   n: number of samples
                   J[i,:,:]: jacobian matrix of the ith sample
        """
        n = thetas.shape[0]
        
        #the calculation of the jacobian for this 6-DOF RRR-RRR manipulator refers to
        #"Modeling and Control of Manipulators, Part I: Geometric and Kinematic Models" by W. Khalil
        
        #construct the kinematic jacobian (base to spherical wrist) first
        #the last link from the spherical wrist to the end effector is not included
        Jk = np.zeros((n,6,6))
        l2 = self.tm.transforms[('link3','link2')][0,3]
        l3 = self.tm.transforms[('link4','link3')][1,3]
        C2 = np.cos(thetas[:,1])
        C3 = np.cos(thetas[:,2])
        C4 = np.cos(thetas[:,3])
        C5 = np.cos(thetas[:,4])
        C6 = np.cos(thetas[:,5])
        S3 = np.sin(thetas[:,2])
        S4 = np.sin(thetas[:,3])
        S5 = np.sin(thetas[:,4])
        S6 = np.sin(thetas[:,5])
        C23 = np.cos(thetas[:,1] + thetas[:,2])
        S23 = np.sin(thetas[:,1] + thetas[:,2])
        
        Jk[:,0,0] = (- C6*C5*S4 - S6*C4)*(S23*l3 - C2*l2)
        Jk[:,1,0] = (S6*C5*S4 - C6*C4)*(S23*l3 - C2*l2)
        Jk[:,2,0] = S5*S4*(S23*l3 - C2*l2)
        Jk[:,3,0] = (C6*C5*C4 - S6*S4)*S23 + C6*S5*C23
        Jk[:,4,0] = (- S6*C5*C4 - C6*S4)*S23 - S6*S5*C23
        Jk[:,5,0] = - S5*C4*S23 + C5*C23
        Jk[:,0,1] = (- C6*C5*C4 + S6*S4)*(l3 - S3*l2) + C6*S5*C3*l2
        Jk[:,1,1] = (S6*C5*C4 + C6*S4)*(l3 - S3*l2) - S6*S5*C3*l2
        Jk[:,2,1] = S5*C4*(l3 - S3*l2) + C5*C3*l2
        Jk[:,3,1] = - C6*C5*S4 - S6*C4
        Jk[:,4,1] = S6*C5*S4 - C6*C4
        Jk[:,5,1] = S5*S4
        Jk[:,0,2] = (- C6*C5*C4 + S6*S4)*l3
        Jk[:,1,2] = (S6*C5*C4 + C6*S4)*l3
        Jk[:,2,2] = S5*C4*l3
        Jk[:,3,2] = - C6*C5*S4 - S6*C4
        Jk[:,4,2] = S6*C5*S4 - C6*C4
        Jk[:,5,2] = S5*S4
        Jk[:,0,3] = 0
        Jk[:,1,3] = 0
        Jk[:,2,3] = 0
        Jk[:,3,3] = C6*S5
        Jk[:,4,3] = - S6*S5
        Jk[:,5,3] = C5
        Jk[:,0,4] = 0
        Jk[:,1,4] = 0
        Jk[:,2,4] = 0
        Jk[:,3,4] = - S6
        Jk[:,4,4] = - C6
        Jk[:,5,4] = 0
        Jk[:,0,5] = 0
        Jk[:,1,5] = 0
        Jk[:,2,5] = 0
        Jk[:,3,5] = 0
        Jk[:,4,5] = 0
        Jk[:,5,5] = 1
        
        #include the relation of the last link by multiplying the kinematic jacobian
        #with a screw transformation matrix
        J = np.zeros((n,6,6))
        for i in range(n):
            screw_matrix = np.zeros((6,6))

            #last transformation is fixed, therefore the following matrix is valid
            #for all angle configurations
            E_T_n = self.tm.get_transform(self.tm.nodes[-2], self.tm.nodes[-1])
            E_R_n = E_T_n[:3,:3]
            E_p_n = E_T_n[:3,3]
            E_P_n_screw = np.array([[0, -E_p_n[2], E_p_n[1]],
                                    [E_p_n[2], 0, -E_p_n[0]],
                                    [-E_p_n[1], E_p_n[0], 0]])
            screw_matrix[:3,:3] = E_R_n
            screw_matrix[-3:,-3:] = E_R_n
            screw_matrix[:3,-3:] = E_P_n_screw @ E_R_n
            
            J[i] = screw_matrix @ Jk[i]
            
            if test:
                self._test_jacobian(J[i], thetas[i,:])
                
        return J
    
    
    def _test_jacobian(self, J, theta_vec):
        """ Checks a jacobian for correctness. This function compares delta_x1 = J(theta) * delta_theta
            and delta_x2 = FK(theta) - FK(theta + delta_theta) for a very small vector delta_theta.
            If delta_x1 and delta_x2 are not close, an exception is thrown.
            Params:
                J: 6x6-matrix: The manipulator jacobian at theta_vec
                theta_vec: 1x6-vector: The joint angles for which the Jacobian is valid
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
                thetas: nx6-matrix or -tensor: angles of the joints (rad)
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
                thetas: nx6-tensor: angles of the joints (rad)
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
        
        #convert pre-filled transformation matrices to tensor
        trans = tf.reshape(tf.convert_to_tensor(self.joint_transformations_notheta[:6], dtype=tf.float64), [1,6,4,4]) #1x6x4x4
        trans_tcp = tf.convert_to_tensor(self.joint_transformations_notheta[6], dtype=tf.float64)                     #4x4
        
        #calculate masks
        Sth = tf.sin(thetas) #nx6
        Cth = tf.cos(thetas) #nx6

        t1 = tf.stack([Cth, -Sth], axis=-1)         #nx6x2
        t2 = tf.stack([Sth, Cth], axis=-1)          #nx6x2
        t3 = tf.stack([Sth, Cth], axis=-1)          #nx6x2
        t4 = tf.ones_like(t3)                       #nx6x2
        t5 = tf.stack([t1, t2, t3, t4], axis=2)     #nx6x4x2
        t6 = tf.ones_like(t5)                       #nx6x4x2
        
        masks = tf.concat([t5, t6], axis=-1)        #nx6x4x4
        
        _i_T_ip1 = tf.multiply(masks, trans)
        _0_T_2 = tf.matmul(_i_T_ip1[:,0,:,:], _i_T_ip1[:,1,:,:])                        #nx4x4
        _2_T_4 = tf.matmul(_i_T_ip1[:,2,:,:], _i_T_ip1[:,3,:,:])                        #nx4x4
        _4_T_6 = tf.reshape(tf.matmul(_i_T_ip1[:,4,:,:], _i_T_ip1[:,5,:,:]), [-1, 4])   #(4n)x4
        _0_T_4 = tf.matmul(_0_T_2, _2_T_4)                                              #nx4x4
        _4_T_tcp = tf.reshape(tf.matmul(_4_T_6, trans_tcp), [-1, 4, 4])                 #nx4x4
        _0_T_tcp = tf.matmul(_0_T_4, _4_T_tcp)                                          #nx4x4
        
        return _0_T_tcp
    
    
    def IK(self, poses):
        """ Analytical Inverse Kinematic Model:
            Transforms end effector poses from 3D cartesian space into the joint space.
            Params:
                poses: nx4x4-matrix: End effector poses in 3D cartesian space
                       n: number of samples
                       poses[i,:,:]: homogenous transformation matrix of the ith sample
                                     representing the end effector pose
            Returns:
                thetas: nx8x6-matrix: All 8 IK solutions
                        first dim:  n samples
                        second dim: 8 solutions
                        third dim:  6 joint angles
        """      
        #rename important variables to improve readability
        n = poses.shape[0]
        l1 = self.tm.transforms[('link2','link1')][2,3]
        l2 = self.tm.transforms[('link3','link2')][0,3]
        l3 = self.tm.transforms[('link4','link3')][1,3]
        l4 = self.tm.transforms[('tcp','link6')][2,3]
        
        #convert poses from Base_T_tcp (input format) to 0_T_6 (the latter has the easier analytical solution)
        #0_T_6 = 0_T_Base * Base_T_tcp * tcp_T_6
        _0_T_base = np.eye(4)
        _0_T_base[2,3] = -l1
        tcp_T_6 = np.eye(4)
        tcp_T_6[2,3] = -l4
        new_poses = np.zeros_like(poses)
        for i in range(n):
            new_poses[i,:,:] = _0_T_base @ poses[i,:,:] @ tcp_T_6
            
        #analytical solution from "Modeling and Control of Manipulators, Part I: Geometric and Kinematic Models" by W. Khalil
        #the maximum number of solutions is eight
        thetas = np.zeros((n,8,6))
        
        sample_times = []
        for i in range(n):
            begin = timer()
            
            #rename important variables to improve readability
            Px = new_poses[i,0,3]
            Py = new_poses[i,1,3]
            Pz = new_poses[i,2,3]
            sx = new_poses[i,0,0]
            sy = new_poses[i,1,0]
            sz = new_poses[i,2,0]
            nx = new_poses[i,0,1]
            ny = new_poses[i,1,1]
            nz = new_poses[i,2,1]
            ax = new_poses[i,0,2]
            ay = new_poses[i,1,2]
            az = new_poses[i,2,2]
            
            #position equations
            #two solutions for the first joint angle
            thetas[i,:4,0] = math.atan2(Py, Px)
            if thetas[i,0,0] < 0:
                thetas[i,4:,0] = thetas[i,:4,0] + math.pi
            else:
                thetas[i,4:,0] = thetas[i,:4,0] - math.pi
            
            #two more solutions for the second joint angle
            for j in range(0,8,4):
                B1 = Px * math.cos(thetas[i,j,0]) + Py * math.sin(thetas[i,j,0])
                X = -2 * Pz * l2
                Y = -2 * B1 * l2
                Z = l3**2 - l2**2 - Pz**2 - B1**2
                sqroot_term = X**2 + Y**2 - Z**2
                if sqroot_term < 0:
                    #in this case, the current solution branch offers no solution
                    thetas[i,j:j+4,:] = np.NaN
                    continue
                sqroot = math.sqrt(sqroot_term)
                denom = X**2 + Y**2
                C2_1 = (Y*Z - X*sqroot) / denom
                S2_1 = (X*Z + Y*sqroot) / denom
                thetas[i,j:j+2,1] = math.atan2(S2_1, C2_1)
                C2_2 = (Y*Z + X*sqroot) / denom
                S2_2 = (X*Z - Y*sqroot) / denom
                thetas[i,j+2:j+4,1] = math.atan2(S2_2, C2_2)
                
                #one solution for the third joint angle
                S3_1 = (- Pz * S2_1 - B1 * C2_1 + l2) / l3
                C3_1 = (- B1 * S2_1 + Pz * C2_1) / l3
                thetas[i,j:j+2,2] = math.atan2(S3_1, C3_1)
                S3_2 = (- Pz * S2_2 - B1 * C2_2 + l2) / l3
                C3_2 = (- B1 * S2_2 + Pz * C2_2) / l3
                thetas[i,j+2:j+4,2] = math.atan2(S3_2, C3_2)
                  
            #orientation equations
            for j in range(0,8,2):
                if np.isnan(thetas[i,j,0]):
                    continue
                #two more solutions for the fourth joint angle
                S1 = math.sin(thetas[i,j,0])
                C1 = math.cos(thetas[i,j,0])
                theta23 = thetas[i,j,1] + thetas[i,j,2]
                S23 = math.sin(theta23)
                C23 = math.cos(theta23)
                Hx = C23 * (C1 * ax + S1 * ay) + S23 * az
                Hz = S1 * ax - C1 * ay
                thetas[i,j,3] = math.atan2(Hz, -Hx)
                if thetas[i,j,3] < 0:
                    thetas[i,j+1,3] = thetas[i,j,3] + math.pi
                else:
                    thetas[i,j+1,3] = thetas[i,j,3] - math.pi
                
                #one solution for the fifth joint angle
                Hy = - S23 * (C1 * ax + S1 * ay) + C23 * az
                S4_1 = math.sin(thetas[i,j,3])
                C4_1 = math.cos(thetas[i,j,3])
                S4_2 = math.sin(thetas[i,j+1,3])
                C4_2 = math.cos(thetas[i,j+1,3])
                S5_1 = S4_1 * Hz - C4_1 * Hx
                S5_2 = S4_2 * Hz - C4_2 * Hx
                C5 = Hy
                thetas[i,j,4] = math.atan2(S5_1, C5)
                thetas[i,j+1,4] = math.atan2(S5_2, C5)
                
                #one solution for the sixth joint angle
                Fx = C23 * (C1 * sx + S1 * sy) + S23 * sz
                Fz = S1 * sx - C1 * sy
                Gx = C23 * (C1 * nx + S1 * ny) + S23 * nz
                Gz = S1 * nx - C1 * ny
                S6_1 = -C4_1 * Fz - S4_1 * Fx
                C6_1 = -C4_1 * Gz - S4_1 * Gx
                S6_2 = -C4_2 * Fz - S4_2 * Fx
                C6_2 = -C4_2 * Gz - S4_2 * Gx
                thetas[i,j,5] = math.atan2(S6_1, C6_1)
                thetas[i,j+1,5] = math.atan2(S6_2, C6_2)
                
            end = timer()
            time_sample = end - begin
            sample_times.append(time_sample)
        
        print("Sum analytical IK: "+str(np.sum(sample_times))+" seconds")
        print("Mean analytical IK: "+str(np.mean(sample_times))+" seconds")
        print("Std analytical IK: "+str(np.std(sample_times))+" seconds")
        
        return thetas


    def plot_configurations(self, joint_configurations):
        """ Plots manipulator configurations in 3D cartesian space.
            Params:
                joint_configurations: nx6-matrix: angles of the joints (rad)
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
                ax = self.tm.plot_frames_in("compi", s=0.05, show_name=False)
            else:
                ax = self.tm.plot_frames_in("compi", s=0.05, ax=ax, show_name=False)
            ax = self.tm.plot_connections_in("compi", ax=ax)
        
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        fig = plt.gcf()
        return fig, ax
   