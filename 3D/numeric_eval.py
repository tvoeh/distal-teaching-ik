import numpy as np
import pickle
from Evaluation3D import Evaluation3D

#evaluates the solutions of the numerical solver TRAC-IK

if __name__ == "__main__":

    joints_num_val = np.load("./data/joints_num_val.npy") #solutions of TRAC-IK
    X_val = np.load("./data/X_val.npy")
    Y_val = np.load("./data/Y_val.npy")
    with open("./data/robot.pkl",'rb') as robotfile:
        robot = pickle.load(robotfile)
    
    #calculate success rate
    n_fails = np.sum(np.isnan(joints_num_val[:,0]))
    success_rate = 1 - (n_fails / joints_num_val.shape[0])
    
    #filter NaN values out before calculating the errors
    notnan = np.invert(np.isnan(joints_num_val[:,0]))
    joints_num_notnan = joints_num_val[notnan]
    X_val_notnan = X_val[notnan]
    Y_val_notnan = Y_val[notnan]

    SE3_num_notnan = robot.FK(joints_num_notnan)
    evl = Evaluation3D(robot, SE3_num_notnan, joints_num_notnan, X_val_notnan, Y_val_notnan)
    pos_error = evl.mae_position()
    orient_error = evl.mae_orientation()
    mean_min_errors, median_min_errors = evl.min_cart_error()
    viol_rate = evl.joint_limit_violation_rate()
    print("\nTRAC-IK success rate: "+str(success_rate)+
          "\nTRAC-IK mean absolute position error of successes: "+str(pos_error)+
          "\nTRAC-IK mean absolute orientation error of successes: "+str(orient_error)+
          "\nJoint limit violation rate: "+str(viol_rate))