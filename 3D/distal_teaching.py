import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from Evaluation3D import Evaluation3D
from helpers import cartesian_loss_wrapper
from timeit import default_timer as timer

np.random.seed(0)

if __name__ == "__main__":
     
    #general parameters
    plot = True
    network_topology = "Atlas"
    #parameters of loss function
    position_weight = 0.75 #weight of the position error
    joint_penalty = False
    penalty_weight = 1
    
    #load data
    with open("./data/robot.pkl",'rb') as robotfile:
        robot = pickle.load(robotfile)
    joint_configurations = np.load("./data/joint_configurations.npy")
    pos_quat = np.load("./data/pos_quat.npy")
    X_train = np.load("./data/X_train.npy")
    Y_train = np.load("./data/Y_train.npy")
    X_val = np.load("./data/X_val.npy")
    SE3_val = np.load("./data/SE3_val.npy")
    Y_val = np.load("./data/Y_val.npy")
    
    n_DOFS = Y_train.shape[1]
    
    if network_topology == "Compi":
        ### suitable architecture for COMPI ###
        model = tf.keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=X_train.shape[1:]),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(2*n_DOFS)])
    else:
        ### suitable architecture for Atlas ###
        model = tf.keras.Sequential([
        layers.Dense(1024, activation='relu', input_shape=X_train.shape[1:]),
        layers.Dense(1024, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(2*n_DOFS)])
            
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss=cartesian_loss_wrapper(robot, position_weight, joint_penalty, penalty_weight),
                  metrics=[])
    
    callbacks = []
    
    begin_train = timer()
    hist = model.fit(X_train, X_train, epochs=1200, batch_size=32, callbacks=callbacks,
                     validation_data=(X_val[:1000], X_val[:1000]))
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
    SE3_ann = robot.FK(joints_ann)
        
    #evaluation
    evl = Evaluation3D(robot, SE3_ann, joints_ann, X_val, Y_val)
    position_errors = evl.get_pos_errors()
    orient_errors = evl.get_ori_errors()
    mean_pos_error = evl.mae_position()
    mean_orient_error = evl.mae_orientation()
    solve_rate = evl.get_solve_rate(1e-2, 3e-2)
    viol_rate = evl.joint_limit_violation_rate()
    print("DT mean absolute position error: "+str(mean_pos_error)+" m"+
          "\nDT mean absolute orientation error: "+str(mean_orient_error)+" rad"+
          "\nJoint limit violation rate: "+str(viol_rate*100)+" %")
    print("Solve rate: "+str(solve_rate*100)+" %")
    
    
    ### plot metrics ### 
    if plot:      
        plt.figure(1)
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.legend(('Loss','Val_Loss'))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')