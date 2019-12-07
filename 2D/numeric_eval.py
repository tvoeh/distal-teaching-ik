import numpy as np
from planar_kinematic_functions import *

#evaluates the solutions of the numerical solver TRAC-IK

if __name__ == "__main__":

    joints_num_val = np.load("./data/joints_num_val.npy") #solutions by TRAC-IK from ubuntu machine
    link_lengths = np.load("./data/link_lengths.npy")
    delta_theta = np.load("./data/delta_theta.npy")
    Y_val = np.load("./data/Y_val.npy")
    joint_limits = np.load("./data/joint_limits.npy")
    X_val = np.load("./data/X_val.npy")
    
    #calculate success rate
    n_fails = np.sum(np.isnan(joints_num_val[:,0]))
    success_rate = 1 - (n_fails / joints_num_val.shape[0])
    
    #filter NaN values out before calculating the errors
    notnan = np.invert(np.isnan(joints_num_val[:,0]))
    joints_num_notnan = joints_num_val[notnan]
    X_val_notnan = X_val[notnan]

    cartesian_num_notnan = FK_2D(joints_num_notnan, link_lengths)
    pos_error = mae_position(X_val_notnan[:,:2], cartesian_num_notnan[:,:2])
    orient_error = mae_orientation_2D(X_val_notnan[:,2], cartesian_num_notnan[:,2])
    mean_min_errors, median_min_errors = min_error_2D_3DOF(Y_val, delta_theta, link_lengths)
    viol_rate = joint_limit_violation_rate(joints_num_notnan, joint_limits)
    print("\nTRAC-IK success rate: "+str(success_rate)+
          "\nTRAC-IK mean absolute position error of successes: "+str(pos_error)+
          "\nTRAC-IK mean absolute orientation error of successes: "+str(orient_error)+
          "\nJoint limit violation rate: "+str(viol_rate))