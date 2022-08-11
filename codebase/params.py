import os
import tensorflow as tf
import numpy as np
# Number of modes
modes = 1
# tt
train_state_select = 0
# Cutoff dimension (number of Fock states)
cutoff_dim = 3
# Input vector length.
input_len = 3
# Number of layers (depth)
Qlayers = 25
# Number of steps in optimization routine performing gradient descent
reps = 200
# Learning rate
lr = 0.05
# Standard deviation of initial parameters
passive_sd = 0.4
active_sd = 0.02
# The gamma parameter in the penalty function, given by the reference paper.
norm_weight = 400
# Seeds for the RNG functions to be able to reproduce results.
tf.random.set_seed(137)
np.random.seed(137)
# Phase space circile restriciton radius.
alpha_clip = 5
save_folder_name = str(input_len) + '_inputs'
os.makedirs(save_folder_name, exist_ok=True)
latent_dim = 2
# Number of iterations to train the classical autoencoder.
classical_epochs = 100
# Number of feedback loop iterations.
sctm_iterations = 3
method_select = 0
fock_space = 0
