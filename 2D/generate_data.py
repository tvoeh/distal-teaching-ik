import numpy as np
import math
from planar_kinematic_functions import *
from matplotlib import pyplot as plt

# constants
PI = math.pi

np.random.seed(552)

if __name__ == "__main__":
    #define manipulator
    link_lengths = np.array([0.5, 0.5, 0.3])
    joint_limits = np.array([[0, PI],[-PI/2, PI/2],[-PI, PI]])
    theta_stds = np.array([1*PI/180, 1*PI/180, 1*PI/180])
    
    train_sampling = "uniform" #possible values: "uniform", "equidistant", "halton" or "cartesian"
    n_samples = 100800
    train_fraction = 800/100800 #fraction of the dataset used for training
    
    validate_singular_configurations = False
    validate_nonsingular_configurations = False
    validate_oow_configurations = False #oow = out of workspace
    validate_trajectory = False
    use_hold_out_area = False
    plot = False
    
    #######################################################################################################
    
    n_DOFS = link_lengths.shape[0]
    
    split = int(train_fraction * n_samples)
    
    joint_configurations = np.zeros((n_samples, n_DOFS))
    for i in range(n_DOFS):
        joint_configurations[:,i] = np.random.uniform(joint_limits[i,0], joint_limits[i,1], n_samples)
        
    assert train_sampling == "uniform" or train_sampling == "equidistant" or train_sampling == "halton" or train_sampling == "cartesian",\
    "Sampling scheme \""+str(train_sampling)+"\" for training data not understood. Choose uniform, equidistant, halton or cartesian."
    if train_sampling == "equidistant":
        #equidistant sampling (in joint space)
        samples_per_joint = math.ceil(split**(1/n_DOFS))
        joint_linspaces = []
        for i in range(n_DOFS):
            joint_linspaces.append(np.linspace(joint_limits[i,0], joint_limits[i,1], samples_per_joint))
            
        joint_grid = np.stack(np.meshgrid(*joint_linspaces))
        joints_train = np.reshape(joint_grid, (n_DOFS, samples_per_joint**n_DOFS)).T
        np.random.shuffle(joints_train)
        joint_configurations[:split] = joints_train[:split]
    elif train_sampling == "halton":
        joints_halton = halton(n_DOFS, split)
        #halton sequence is in the interval [0,1] and needs to be scaled by the joint intervals and shifted
        joint_intervals = []
        joint_minimums = []
        for i in range(n_DOFS):
            joint_intervals.append(joint_limits[i,1] - joint_limits[i,0])
            joint_minimums.append(joint_limits[i,0])
        scaled_shifted_joints_halton = joints_halton * joint_intervals + joint_minimums
        joint_configurations[:split] = scaled_shifted_joints_halton
    elif train_sampling == "cartesian":
        #arbitrarily chosen subspace of the maximum workspace (for no joint limits) where we train and test
        x_interval = [-0.5, 0.5]
        y_interval = [0.5, 1]
        alpha_interval = [1, 2]
        
        #we sample in cartesian space, so we have no labels at hand
        joint_configurations[:,:] = np.nan
        
        #uniform sampling in cartesian space
        points_cartesian = np.zeros((n_samples, 3))
        points_cartesian[:,0] = np.random.uniform(x_interval[0], x_interval[1], n_samples)
        points_cartesian[:,1] = np.random.uniform(y_interval[0], y_interval[1], n_samples)
        points_cartesian[:,2] = np.random.uniform(alpha_interval[0], alpha_interval[1], n_samples)
        
    
    #if specified, set test data to singular configurations only (theta2=0 or theta2=pi)
    #only valid for 3 DOF RRR manipulator
    if validate_singular_configurations:
        joint_configurations[split:,1] = 0
    
    #if specified, set test data to non-singular configurations only ( abs(det(J)) >> 0 )
    #the half of the test data with the highest values of abs(det(J)) is kept
    #the other half is removed from the test data
    if validate_nonsingular_configurations:
        val_data_before = joint_configurations[split:]
        J = jacobian_2D_3DOF(val_data_before, link_lengths)
        dets = np.linalg.det(J)
        args_dets_sorted = np.argsort(np.abs(dets))
        val_data_after = val_data_before[args_dets_sorted[int(args_dets_sorted.shape[0]/2):]]
        joint_configurations = np.vstack((joint_configurations[:split], val_data_after))
    
    if not train_sampling == "cartesian":
        #transform the samples via FK to cartesian space
        points_cartesian = FK_2D(joint_configurations, link_lengths)
        
    #if specified, train on one half of the workspace, test on the other
    if use_hold_out_area:        
        train_bools = points_cartesian[:,0] > 0
        assert np.sum(train_bools) > split
        
        #extract training data
        points_cartesian_train = points_cartesian[train_bools]
        points_cartesian_train = points_cartesian_train[:split]
        joint_configurations_train = joint_configurations[train_bools]
        joint_configurations_train = joint_configurations_train[:split]
        
        #extract test data
        lower_x_bound = -0.85
        upper_x_bound = -0.75
        val_bools = np.logical_and(points_cartesian[:,0] > lower_x_bound, points_cartesian[:,0] < upper_x_bound)
        points_cartesian_val = points_cartesian[val_bools]
        joint_configurations_val = joint_configurations[val_bools]

        #concatenate train and test data
        points_cartesian = np.vstack((points_cartesian_train, points_cartesian_val))
        joint_configurations = np.vstack((joint_configurations_train, joint_configurations_val))
              
    #if specified, set test data to unreachable poses only
    if validate_oow_configurations:
        joint_configurations[split:,:] = np.NaN
        total_length = np.sum(link_lengths)
        n_val_samples = n_samples - split
        magnitudes = np.random.uniform(2*total_length, 3*total_length, n_val_samples)
        directions = np.random.uniform(-PI, PI, n_val_samples)
        points_cartesian[split:,0] = np.cos(directions) * magnitudes
        points_cartesian[split:,1] = np.sin(directions) * magnitudes
    
    #if specified, set test data to a simple trajectory
    if validate_trajectory:
        n = 50
        start_x = -0.5
        end_x = 0.5
        start_y = 1
        end_y = 1
        alpha = PI/2
        x = np.linspace(start_x, end_x, n)
        y = np.linspace(start_y, end_y, n)
        points_val = np.stack((x, y, np.zeros(n)+alpha )).T
        points_cartesian = np.vstack((points_cartesian[:split], points_val))
        joints_val1, joints_val2 = IK_2D_3DOF(points_val, link_lengths)
        joint_configurations = np.vstack((joint_configurations[:split], joints_val1))
        
        
    #split data into training and test set     
    X_train = points_cartesian[:split]
    Y_train = joint_configurations[:split]
    X_val = points_cartesian[split:]
    Y_val = joint_configurations[split:]
    
    if plot:
        plt.figure(1)
        plt.scatter(X_val[:,0], X_val[:,1], s=12, label='Test')
        plt.scatter(X_train[:,0], X_train[:,1], s=12, label='Train')
        if validate_oow_configurations:
            plt.xlim([-5,5])
            plt.ylim([-5,5])
        else:
            plt.xlim([-1.5,1.5])
            plt.ylim([-1.2,1.8])
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.legend()
        plt.grid(True)
        fig=plt.gcf()
        fig.set_size_inches(4, 4)
        fig.savefig('./plots/neu.pdf', papertype='a4', orientation='portrait', bbox_inches="tight")
    
    #save generated data
    np.save("./data/link_lengths.npy", link_lengths)
    np.save("./data/joint_limits.npy", joint_limits)
    np.save("./data/delta_theta.npy", theta_stds)
    np.save("./data/joint_configurations.npy", joint_configurations)
    np.save("./data/points_cartesian.npy", points_cartesian)
    np.save("./data/X_train.npy", X_train)
    np.save("./data/Y_train.npy", Y_train)
    np.save("./data/X_val.npy", X_val)
    np.save("./data/Y_val.npy", Y_val)
    
    print("Data generated and saved.")