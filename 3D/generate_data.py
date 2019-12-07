import numpy as np
import math
import pickle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Compi import Compi
from Atlas import Atlas
from helpers import quaternion_from_matrix, halton

np.random.seed(0)

if __name__ == "__main__":
    robot = Atlas("./urdf/Atlas.urdf") #initialize robot parameters via urdf file
    #alternative: robot = Compi("./urdf/Compi.urdf")
    
    train_sampling = "uniform" #possible values: "uniform", "equidistant", "halton" or "cartesian"
    n_samples = 132000
    train_fraction = 32/132 #fraction of the dataset used for training
    
    validate_singular_configurations = False
    validate_nonsingular_configurations = False
    validate_oow_configurations = False #oow = out of workspace
    validate_trajectory = False
    use_hold_out_area = False
    plot = False
        
    #######################################################################################################

    n_DOFS = robot.n_DOFS
    split = int(train_fraction * n_samples)
        
    joint_configurations = np.zeros((n_samples, n_DOFS))
    for i, joint in enumerate(robot.sorted_joint_names):
        #uniform sampling between lower and upper joint limit
        joint_configurations[:,i] = np.random.uniform(robot.joint_info[joint][4][0],
                                                      robot.joint_info[joint][4][1], n_samples)
    
    assert train_sampling == "uniform" or train_sampling == "equidistant" or train_sampling == "halton" or train_sampling == "cartesian",\
    "Sampling scheme \""+str(train_sampling)+"\" for training data not understood. Choose uniform, equidistant, halton or cartesian."
    if train_sampling == "equidistant":
        #equidistant sampling (in joint space)
        samples_per_joint = math.ceil(split**(1/n_DOFS))
        joint_linspaces = []
        for i, joint in enumerate(robot.sorted_joint_names):
            joint_linspaces.append(np.linspace(robot.joint_info[joint][4][0],
                                               robot.joint_info[joint][4][1], samples_per_joint))
            
        joint_grid = np.stack(np.meshgrid(*joint_linspaces))
        joints_train = np.reshape(joint_grid, (n_DOFS, samples_per_joint**n_DOFS)).T
        np.random.shuffle(joints_train)
        joint_configurations[:split] = joints_train[:split]
    elif train_sampling == "halton":
        joints_halton = halton(n_DOFS, split)
        #halton sequence is in the interval [0,1] and needs to be scaled by the joint intervals and shifted
        joint_intervals = []
        joint_minimums = []
        for i, joint in enumerate(robot.sorted_joint_names):
            joint_intervals.append(robot.joint_info[joint][4][1] - robot.joint_info[joint][4][0])
            joint_minimums.append(robot.joint_info[joint][4][0])
        scaled_shifted_joints_halton = joints_halton * joint_intervals + joint_minimums
        joint_configurations[:split] = scaled_shifted_joints_halton
    elif train_sampling == "cartesian":
        #arbitrarily chosen subspace of the maximum workspace (for no joint limits) where we train and test
        #it is secured via numerical tests that this set really lies in the workspace
        subspace = robot.subspace
        
        #we sample in cartesian space, so we have no labels at hand
        joint_configurations[:,:] = np.nan
        
        #uniform sampling in cartesian space
        pos_quat = np.zeros((n_samples, 7))
        for i in range(pos_quat.shape[1]):
            pos_quat[:,i] = np.random.uniform(subspace[i,0], subspace[i,1], n_samples)
        #normalize quaternions
        magnitudes = np.reshape(np.linalg.norm(pos_quat[:,3:], axis=1), (-1,1))
        pos_quat[:,3:] /= magnitudes
        
        SE3 = np.zeros((n_samples, 4, 4))
        SE3[:,:,:] = np.nan
        
            
    #if specified, set test data to singular configurations only
    #Remark: 15-DOF is not supported in this script. The data was generated separately with ROS and PyKDL (Python 2.7) for the paper.
    if validate_singular_configurations:
        assert robot.has_singularity_condition, "The kinematic chain needs to have a singularity condition in joint space."
        joint_configurations[split:] = robot.turn_configurations_singular(joint_configurations[split:])
        
    #if specified, set test data to non-singular configurations only ( abs(det(J)) >> 0 )
    #the half of the test data with the highest values of abs(det(J)) is kept
    #the other half is removed from the test data
    #Remark: 15-DOF is not supported in this script. The data was generated separately with ROS and PyKDL (Python 2.7) for the paper.
    if validate_nonsingular_configurations:
        val_data_before = joint_configurations[split:]
        assert robot.has_jacobian, "The kinematic chain needs to have a Jacobian."
        J = robot.jacobian(val_data_before)
        dets = np.linalg.det(J)
        args_dets_sorted = np.argsort(np.abs(dets))
        val_data_after = val_data_before[args_dets_sorted[int(args_dets_sorted.shape[0]/2):]]
        joint_configurations = np.vstack((joint_configurations[:split], val_data_after))
    
    if not train_sampling == "cartesian":
        #transform the samples via FK to cartesian space
        SE3 = robot.FK(joint_configurations)
    
        #convert rotation matrices to quaternions
        #one sample shall be a pair {7-dim vector(x, y, z, 4-dim-quaternion), n_DOFS-dim vector(joint angles)}
        pos_quat = np.zeros((joint_configurations.shape[0], 7))
        pos_quat[:,:3] = SE3[:,:3,3]
        pos_quat[:,3:] = quaternion_from_matrix(SE3[:,:3,:3])
    
    #if specified, train on one half of the workspace, test on the other
    if use_hold_out_area:        
        train_bools = pos_quat[:,0] > 0
        assert np.sum(train_bools) > split, "Not enough samples to extract the desired number of training samples. Increase the number of samples."
        
        #extract training data
        pos_quat_train = pos_quat[train_bools]
        pos_quat_train = pos_quat_train[:split]
        SE3_train = SE3[train_bools]
        SE3_train = SE3_train[:split]
        joint_configurations_train = joint_configurations[train_bools]
        joint_configurations_train = joint_configurations_train[:split]
        
        #extract test data
        lower_x_bound = -0.85
        upper_x_bound = -0.75
        val_bools = np.logical_and(pos_quat[:,0] > lower_x_bound, pos_quat[:,0] < upper_x_bound)
        pos_quat_val = pos_quat[val_bools]
        SE3_val = SE3[val_bools]
        joint_configurations_val = joint_configurations[val_bools]

        #concatenate train and test data
        pos_quat = np.vstack((pos_quat_train, pos_quat_val))
        SE3 = np.vstack((SE3_train, SE3_val))
        joint_configurations = np.vstack((joint_configurations_train, joint_configurations_val))
        
    #if specified, set test data to unreachable poses only
    if validate_oow_configurations:
        joint_configurations[split:,:] = np.NaN
        total_length = 0
        for joint in robot.joint_info:
                total_length += np.linalg.norm(robot.joint_info[joint][2][:3,3])
        n_val_samples = n_samples - split
        
        magnitudes = np.random.uniform(2*total_length, 3*total_length, n_val_samples)
        vec_x = np.random.normal(size = n_val_samples)
        vec_y = np.random.normal(size = n_val_samples)
        vec_z = np.random.normal(size = n_val_samples)
        vec = np.stack((vec_x, vec_y, vec_z)).T
        vec = vec / np.reshape(np.linalg.norm(vec, axis=1), (-1, 1))
        assert not np.any(np.isinf(vec))
        positions_val = vec * np.reshape(magnitudes, (-1, 1))
        pos_quat[split:,:3] = positions_val
        SE3[split:,:3,3] = positions_val
        
        if plot:
            fig = plt.figure(2)
            ax = Axes3D(fig)
            ax.scatter(pos_quat[split:, 0], pos_quat[split:, 1], pos_quat[split:, 2], label='Test')
            ax.scatter(pos_quat[:split, 0], pos_quat[:split, 1], pos_quat[:split, 2], label='Train')
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_zlabel("z [m]")
            ax.legend()
        
    #if specified, set test data to a simple trajectory
    if validate_trajectory:
        n = 50
        start_x = robot.trajectory[0]
        end_x = robot.trajectory[1]
        start_y = robot.trajectory[2]
        end_y = robot.trajectory[3]
        start_z = robot.trajectory[4]
        end_z = robot.trajectory[5]
        x = np.linspace(start_x, end_x, n)
        y = np.linspace(start_y, end_y, n)
        z = np.linspace(start_z, end_z, n)
        SE3_val = np.reshape(np.eye(4), (1,4,4))
        SE3_val = np.repeat(SE3_val, n, axis=0)
        SE3_val[:,0,3] = x
        SE3_val[:,1,3] = y
        SE3_val[:,2,3] = z
        SE3 = np.vstack((SE3[:split], SE3_val))
        quat_val = quaternion_from_matrix(SE3_val[:,:3,:3])
        pos_val = np.stack((x, y, z)).T
        pos_quat_val = np.hstack((pos_val, quat_val))
        pos_quat = np.vstack((pos_quat[:split], pos_quat_val))
        if robot.has_IK:
            joints_val_ik = robot.IK(SE3_val)
            joint_configurations = np.vstack((joint_configurations[:split], joints_val_ik[:, 0, :]))
        else:
            joints_val = np.zeros((n, robot.n_DOFS))
            joints_val[:,:] = np.nan
            joint_configurations = np.vstack((joint_configurations[:split], joints_val))
            
    #split data into training and test set 
    X_train = pos_quat[:split]
    SE3_train = SE3[:split]
    Y_train = joint_configurations[:split]
    X_val = pos_quat[split:]
    SE3_val = SE3[split:]
    Y_val = joint_configurations[split:]
    
    if plot:
        plt.figure(1)
        plt.scatter(X_val[:,0], X_val[:,2], label='Test')
        plt.scatter(X_train[:,0], X_train[:,2], label='Train')
        if validate_oow_configurations:
            plt.xlim([-5,5])
            plt.ylim([-5,5])
        else:
            plt.xlim([-2.5,2.5])
            plt.ylim([-2.5,2.5])
        plt.xlabel("x [m]")
        plt.ylabel("z [m]")
        plt.legend()
    
    #save generated data
    with open("./data/robot.pkl",'wb') as robotfile:
        pickle.dump(robot, robotfile)
    np.save("./data/joint_configurations.npy", joint_configurations)
    np.save("./data/pos_quat.npy", pos_quat)
    np.save("./data/X_train.npy", X_train)
    np.save("./data/SE3_train.npy", SE3_train)
    np.save("./data/Y_train.npy", Y_train)
    np.save("./data/X_val.npy", X_val)
    np.save("./data/SE3_val.npy", SE3_val)
    np.save("./data/Y_val.npy", Y_val)
    
    print("Data generated and saved.")