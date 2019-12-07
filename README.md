# Description

This repository contains the main source code to perform a detailed evaluation of distal teaching, a machine learning approach to solve the inverse kinematics problem, under known forward kinematics.
For three serial and rigid mechanisms of different complexity (3-DoF (planar), 6-DoF and 15-DoF), the provided code enables to train and evaluate an inverse kinematic model via distal teaching using [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/).
Further, analytical solutions are provided in native Python for the 3-DoF and 6-DoF mechanisms and the numerical solver [TRAC-IK](http://wiki.ros.org/trac_ik) can be used for all three mechanisms as a comparison.
The results in *"Comparison of Distal Teacher Learning with Numerical and Analytical Methods to Solve Inverse Kinematics for Rigid-Body Mechanisms"* are based on this implementation.

# Folder Structure
- The folder *2D/* contains a very basic implementation of distal teaching specifically for planar arms.
- The folder *3D/* contains a more general implementation of distal teaching suited for the three-dimensional case.

# Requirements
For training and evaluating an inverse kinematic model via distal teaching, Python 3 is mandatory. The required libraries can be found in *requirements_distal_teaching.txt*.

The numerical solver TRAC-IK depends on ROS which works best with Python 2.7.
If you want to solve the inverse kinematic problem for query poses with TRAC-IK via the script *solve_with_trac_ik.py*, install *trac_ik_python* by running `sudo apt-get install ros-melodic-trac-ik` (if your distribution differs, insert it instead of "melodic").
Before you run *solve_with_trac_ik.py*, make sure to use `roslaunch` with a launch file and a fitting urdf file (subfolder *2D/urdf/* or *3D/urdf/*) as an argument. As an example launch file, *example.launch* is given.

# Basic usage
First of all, training and testing data has to be generated. To do so, run the script *generate_data.py* (from folder *2D/* or *3D/*) that stores the datasets in the subfolder *data/*. At the top of the script *generate_data.py*, there are a few modifiable parameters.
After storing training and testing data, an inverse kinematic model (feed-forward neural network) can be trained by running *distal_teaching.py*. As soon as the training is finished, the model will be stored in the subfolder *models/*. Furthermore, some basic evaluation metrics are printed after the training.

As a comparison, the generated test dataset can also be solved using the numerical solver TRAC-IK by running *solve_with_trac_ik.py*. Note that this requires a launched urdf file via `roslaunch`.
An evaluation of the solutions by TRAC-IK can be performed by running *numeric_eval.py*.