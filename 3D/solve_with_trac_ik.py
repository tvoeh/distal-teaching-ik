#!/usr/bin/env python

import numpy as np
from trac_ik_python.trac_ik import IK
from timeit import default_timer as timer

#Intended for use under ROS. The urdf file of the robot has to be launched via roslaunch beforehand.

use_last_solution_as_seed = False
robot = "Atlas"
#error thresholds
pos_error = 5e-3
ori_error = 1.5e-2
#timeout
timeout = 0.005

np.random.seed(246)

X_val = np.load("data/X_val.npy")

if robot == "Atlas":
    ik_solver = IK("l_foot", "tcp", timeout=timeout) #start and end of kinematic chain
    limits =    [[-0.436, 0.436],
                [-0.698, 0.698],
                [0.0, 2.45],
                [-1.75, 0.524],
                [-0.47, 0.495],
                [-0.32, 1.14],
                [-0.610865, 0.610865],
                [-1.2, 1.28],
                [-0.790809, 0.790809],
                [-1.9635, 1.9635],
                [-1.39626, 1.74533],
                [0.0, 3.14159],
                [0.0, 2.35619],
                [-1.571, 1.571],
                [-0.436, 1.571]]
else:
    ik_solver = IK("base_link", "tcp", timeout=timeout) #start and end of kinematic chain
    limits =    [[-2.094, 2.094],
                [0.17079632679489665, 2.9707963267948965],
                [-3.140796326794897, -0.0007963267948964958],
                [-2.1, 2.1],
                [-2.1, 2.1],
                [-2.1, 2.1]]


#initial configurations are random
seed_configurations = np.zeros((X_val.shape[0], ik_solver.number_of_joints))
ind = 0
for lower, upper in limits:
    seed_configurations[:,ind] = np.random.uniform(lower, upper, X_val.shape[0])
    ind += 1
    
#first seed is zero-vector for trajectory evaluation
if use_last_solution_as_seed:
    seed_state = np.zeros_like(seed_configurations[0,:])
   
#solve for validation data
joints_num_val = np.zeros((X_val.shape[0], ik_solver.number_of_joints))

sample_times = []
for i in range(joints_num_val.shape[0]):
    if not use_last_solution_as_seed:
        seed_state = seed_configurations[i,:]
    begin = timer()
    joints_i = ik_solver.get_ik(seed_state, #initial configuration
                    X_val[i,0], X_val[i,1], X_val[i,2], #desired position (x, y, z)
                    X_val[i,4], X_val[i,5], X_val[i,6], X_val[i,3], #desired orientation (quaternions: xq, yq, zq, wq)
                    pos_error, pos_error, pos_error, #allowed position error (x, y, z)
                    ori_error, ori_error, ori_error) #allowed orientation error (rotation over axis x, y, z)
    end = timer()
    sample_times.append(end - begin)
    joints_num_val[i,:] = joints_i
    if use_last_solution_as_seed and not np.any(np.isnan(joints_num_val[i,:])):
        seed_state = joints_num_val[i,:] #for trajectory evaluation: seed is always the current configuration (last solution)
print("Sum TRAC-IK: "+str(np.sum(sample_times))+" seconds")
print("Mean TRAC-IK: "+str(np.mean(sample_times))+" seconds")
print("Std TRAC-IK: "+str(np.std(sample_times))+" seconds")

np.save("data/joints_num_val.npy", joints_num_val)
