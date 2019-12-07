#!/usr/bin/env python

import math
import numpy as np
from trac_ik_python.trac_ik import IK
from timeit import default_timer as timer

#Intended for use under ROS. The urdf file of the robot has to be launched via roslaunch beforehand.

use_last_solution_as_seed = False
#error thresholds
pos_error = 5e-3
ori_error = 1.5e-2
#timeout
timeout = 0.005

np.random.seed(246)

X_val = np.load("data/X_val.npy")

ik_solver = IK("base_link", "tcp", timeout=timeout) #start and end of kinematic chain
limits =    [[0, math.pi],
            [-math.pi/2, math.pi/2],
            [-math.pi, math.pi]]

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
				    X_val[i,0], X_val[i,1], 0.0, #desired position (x, y, z)
				    0.0, 0.0, math.sin(X_val[i,2]/2), math.cos(X_val[i,2]/2), #desired orientation (quaternions: xq, yq, zq, wq)
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
