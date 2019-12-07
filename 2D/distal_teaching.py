import math
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from planar_kinematic_functions import *
from timeit import default_timer as timer
import os

# constants
PI = math.pi

np.random.seed(0)


if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #training on CPU (faster here)
    
    plot = True
    #parameters of loss function
    pos_weight = 0.75 #weight of the position error
    joint_penalty = False
    penalty_weight = 1
    
    #load data
    link_lengths = np.load("./data/link_lengths.npy")
    joint_limits = np.load("./data/joint_limits.npy")
    delta_theta = np.load("./data/delta_theta.npy")
    joint_configurations = np.load("./data/joint_configurations.npy")
    points_cartesian = np.load("./data/points_cartesian.npy")
    X_train = np.load("./data/X_train.npy")
    Y_train = np.load("./data/Y_train.npy")
    X_val = np.load("./data/X_val.npy")
    Y_val = np.load("./data/Y_val.npy")
    
    n_DOFS = link_lengths.shape[0]
    
    #transform orientation to sin/cos (smooth)
    sin = np.atleast_2d(np.sin(points_cartesian[:,2])).T
    cos = np.atleast_2d(np.cos(points_cartesian[:,2])).T
    points_cartesian_sincos = np.hstack((points_cartesian[:,:2], sin, cos))
    
    X_val_unmodified = X_val.copy() #for error calculation later on
    
    split = X_train.shape[0]
    X_train = points_cartesian_sincos[:split]
    X_val = points_cartesian_sincos[split:]
        
    model = tf.keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=X_train.shape[1:]),
    layers.Dense(256, activation='relu'),
    layers.Dense(2*n_DOFS)])
        
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss=cartesian_loss_sincos(link_lengths, pos_weight, joint_limits, joint_penalty, penalty_weight),
                  metrics=[])
    
    callbacks = []
    
    begin_train = timer()
    hist = model.fit(X_train, X_train, epochs=4000, batch_size=32, callbacks=callbacks,
                     validation_data=(X_val[:128], X_val[:128]))
    end_train = timer()
    print("Runtime Training: "+str(end_train - begin_train)+" seconds")
    
    model.save("./models/DT.h5")
    
    begin = timer()
    joints_sincos_ann = model.predict(X_val)
    
    #calculate the joint angles from sin/cos representation
    j = []
    for i in range(n_DOFS):
        j.append(np.atleast_2d(np.arctan2(joints_sincos_ann[:,i], joints_sincos_ann[:,i+n_DOFS])).T)
    joints_ann = np.hstack(j)
    
    end = timer()
    print("Runtime Prediction: "+str(end - begin)+" seconds")
    
    #propagate predictions through the forward model
    cartesian_ann = FK_2D(joints_ann, link_lengths)
    
    #evaluation
    pos_error = mae_position(X_val_unmodified[:,:2], cartesian_ann[:,:2])
    orient_error = mae_orientation_2D(X_val_unmodified[:,2], cartesian_ann[:,2])
    solve_rate = get_solve_rate(X_val_unmodified, cartesian_ann, 1e-2, 3e-2)
    viol_rate = joint_limit_violation_rate(joints_ann, joint_limits)
    print("Distal teaching (ANN) mean absolute position error: "+str(pos_error)+" m"+
          "\nDistal teaching (ANN) mean absolute orientation error: "+str(orient_error)+" rad"+
          "\nJoint limit violation rate: "+str(viol_rate*100)+" %")
    print("Solve rate: "+str(solve_rate*100)+" %")
    
    #calculate IK solutions in case of further comparisons
    ik1, ik2 = IK_2D_3DOF(points_cartesian, link_lengths)
    
    ### plot metrics ### 
    if plot:      
        plt.figure()
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.legend(('Loss','Val_Loss'))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')