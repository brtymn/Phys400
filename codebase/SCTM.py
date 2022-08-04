#!/usr/bin/env python
# coding: utf-8

# # Self Consistent Training Model For The Hybrid Autoencoder

# ### Library imports.

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from sklearn.metrics import accuracy_score, precision_score, recall_score

import matplotlib.pyplot as plt
import numpy as np
import math
import os

import strawberryfields as sf
from strawberryfields import ops
sf.about()

from matplotlib.pyplot import figure
figure(figsize=(12, 6), dpi=100)



# ### Initialization of layer weights.

# In[4]:




# In[5]:


autoencoder = Autoencoder(latent_dim)



x_train, target_state = GenerateTargetState(input_len, train_state_select)
print('The target state for the training is chosen to be ' + str(target_state))


# ### Compile the classical autoencoder.

# In[7]:


autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())


# ### Initialization of the feedback loop that contains the training of the classical and quantum layers.

# In[8]:


fid_progress = [[], [], [], [], [], [], []]
loss_progress = [[], [], [], [], [], [], []]


for j in range(sctm_iterations):
    history = autoencoder.fit(x_train, x_train,epochs=classical_epochs)

    encoded_st = autoencoder.encoder(np.array([target_state])).numpy()
    #decoded_st = autoencoder.decoder(encoded_st).numpy()

    def layer(params, q):
        """CV quantum neural network layer acting on ``N`` modes.

        Args:
            params (list[float]): list of length ``2*(max(1, N-1) + N**2 + n)`` containing
                the number of parameters for the layer
            q (list[RegRef]): list of Strawberry Fields quantum registers the layer
                is to be applied to
        """
        ops.Dgate(tf.clip_by_value(encoded_st[0][0], clip_value_min = -alpha_clip, clip_value_max = alpha_clip), math.degrees(encoded_st[0][1])) | q[0]

        N = len(q)
        M = int(N * (N - 1)) + max(1, N - 1)

        rphi = params[-N+1:]
        s = params[M:M+N]
        dr = params[2*M+N:2*M+2*N]
        dp = params[2*M+2*N:2*M+3*N]
        k = params[2*M+3*N:2*M+4*N]

        ops.Rgate(rphi[0]) | q[0]

        for i in range(N):
            ops.Sgate(s[i]) | q[i]

        ops.Rgate(rphi[0]) | q[0]

        for i in range(N):
            ops.Dgate(dr[i], dp[i]) | q[i]
            ops.Kgate(k[i]) | q[i]

    # initialize engine and program
    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim})
    qnn = sf.Program(modes)

    # initialize QNN weights
    weights = init_weights(modes, Qlayers) # our TensorFlow weights
    num_params = np.prod(weights.shape)   # total number of parameters in our model

    # Create array of Strawberry Fields symbolic gate arguments, matching
    # the size of the weights Variable.
    sf_params = np.arange(num_params).reshape(weights.shape).astype(np.str)
    sf_params = np.array([qnn.params(*i) for i in sf_params])


    # Construct the symbolic Strawberry Fields program by
    # looping and applying layers to the program.
    with qnn.context as q:
        for k in range(Qlayers):
            layer(sf_params[k], q)

    def cost(weights):
        # Create a dictionary mapping from the names of the Strawberry Fields
        # free parameters to the TensorFlow weight values.
        mapping = {p.name: w for p, w in zip(sf_params.flatten(), tf.reshape(weights, [-1]))}

        # Run engine
        state = eng.run(qnn, args=mapping).state

        # Extract the statevector
        ket = state.ket()

        # Compute the fidelity between the output statevector
        # and the target state.
        fidelity = tf.abs(tf.reduce_sum(tf.math.conj(ket) * target_state)) ** 2

        # Objective function to minimize
        #cost = tf.abs(tf.reduce_sum(tf.math.conj(ket) * target_state) - 1)
        #return cost, fidelity, ket
        # Instead of the Cost function, maybe it is better to break it down to components
        # at least, when the Fock basis is insufficent, it will be visible
        difference = tf.reduce_sum(tf.abs(ket - target_state))
        fidelity = tf.abs(tf.reduce_sum(tf.math.conj(ket) * target_state)) ** 2
        return difference, fidelity, ket, tf.math.real(state.trace())

    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    fidp = []
    lossp = []
    best_fid = 0

    for i in range(reps):
        # reset the engine if it has already been executed
        if eng.run_progs:
            eng.reset()

        with tf.GradientTape() as tape:
            loss, fid, ket, trace = cost(weights)

        # Stores fidelity at each step
        fidp.append(fid.numpy())
        lossp.append(loss)

        if fid > best_fid:
            # store the new best fidelity and best state
            best_fid = fid.numpy()
            learnt_state = ket.numpy()

        # one repetition of the optimization
        gradients = tape.gradient(loss, weights)
        opt.apply_gradients(zip([gradients], [weights]))
         # Prints progress at every rep
        if i % 1 == 0:
            print("Rep: {} Cost: {:.4f} Fidelity: {:.4f} Trace: {:.4f}".format(i, loss, fid, trace))

    fid_progress[j] = fidp
    loss_progress[j] = lossp

    x_train = np.array([learnt_state])


# ### Produce fidelity plot.

# In[9]:


fig, ax = plt.subplots(figsize=(12, 12), dpi = 100)

# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(16)

for i in range(sctm_iterations):
    plt.plot(fid_progress[i], label = "Fidelity of iteration " + str(i), linewidth = 3)

plt.ylabel("Fidelity", fontsize = '20')
ax.yaxis.set_ticks(np.arange(0, 1, 0.4))
plt.xlabel("Epoch", fontsize = '20')
ax.xaxis.set_ticks(np.arange(0, reps, 40))
plt.title('SCTM Fidelity', fontsize = '24')
plt.legend()
plt.savefig('SCTM_fidelity.png')


# ### Produce loss plot.

# In[10]:


fig, ax = plt.subplots(figsize=(12, 12), dpi = 100)

# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(16)

for i in range(sctm_iterations):
    plt.plot(loss_progress[i], label = "Training loss of iteration " + str(i), linewidth = 3)

plt.ylabel("Training Loss", fontsize = '20')
ax.yaxis.set_ticks(np.arange(0, 1, 0.4))
plt.xlabel("Epoch", fontsize = '20')
ax.xaxis.set_ticks(np.arange(0, reps, 40))
plt.title('SCTM Training Loss', fontsize = '24')
plt.legend()
plt.savefig('SCTM_loss.png')


# ### Definition of the function that plots Wigner functions.

# In[11]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def wigner(rho):
    """This code is a modified version of the 'iterative' method
    of the wigner function provided in QuTiP, which is released
    under the BSD license, with the following copyright notice:

    Copyright (C) 2011 and later, P.D. Nation, J.R. Johansson,
    A.J.G. Pitchford, C. Granade, and A.L. Grimsmo.

    All rights reserved."""
    import copy

    # Domain parameter for Wigner function plots
    l = 5.0
    cutoff = rho.shape[0]

    # Creates 2D grid for Wigner function plots
    x = np.linspace(-l, l, 100)
    p = np.linspace(-l, l, 100)

    Q, P = np.meshgrid(x, p)
    A = (Q + P * 1.0j) / (2 * np.sqrt(2 / 2))

    Wlist = np.array([np.zeros(np.shape(A), dtype=complex) for k in range(cutoff)])

    # Wigner function for |0><0|
    Wlist[0] = np.exp(-2.0 * np.abs(A) ** 2) / np.pi

    # W = rho(0,0)W(|0><0|)
    W = np.real(rho[0, 0]) * np.real(Wlist[0])

    for n in range(1, cutoff):
        Wlist[n] = (2.0 * A * Wlist[n - 1]) / np.sqrt(n)
        W += 2 * np.real(rho[0, n] * Wlist[n])

    for m in range(1, cutoff):
        temp = copy.copy(Wlist[m])
        # Wlist[m] = Wigner function for |m><m|
        Wlist[m] = (2 * np.conj(A) * temp - np.sqrt(m) * Wlist[m - 1]) / np.sqrt(m)

        # W += rho(m,m)W(|m><m|)
        W += np.real(rho[m, m] * Wlist[m])

        for n in range(m + 1, cutoff):
            temp2 = (2 * A * Wlist[n - 1] - np.sqrt(m) * temp) / np.sqrt(n)
            temp = copy.copy(Wlist[n])
            # Wlist[n] = Wigner function for |m><n|
            Wlist[n] = temp2

            # W += rho(m,n)W(|m><n|) + rho(n,m)W(|n><m|)
            W += 2 * np.real(rho[m, n] * Wlist[n])

    return Q, P, W / 2


# ### Obtain the target and learnt states from the quantum decoder.

# In[12]:


rho_target = np.outer(target_state, target_state.conj())
rho_learnt = np.outer(learnt_state, learnt_state.conj())


# ### Plot the target state as a Wigner function.

# In[13]:


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
X, P, W = wigner(rho_target)
ax.plot_surface(X, P, W, cmap="Spectral", lw=0.5, rstride=1, cstride=1)
ax.contour(X, P, W, 10, cmap="Spectral", linestyles="solid", offset=-0.17)
ax.set_axis_off()
fig.show()


# ### Plot the learnt state as a Wigner function.

# In[14]:


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
X, P, W = wigner(rho_learnt)
ax.plot_surface(X, P, W, cmap="Spectral", lw=0.5, rstride=1, cstride=1)
ax.contour(X, P, W, 10, cmap="Spectral", linestyles="solid", offset=-0.17)
ax.set_axis_off()
fig.show()


# In[15]:


np.savetxt(save_folder_name + '/x' + str(train_state_select) + '.txt', X)
np.savetxt(save_folder_name + '/p' + str(train_state_select) + '.txt', P)
np.savetxt(save_folder_name + '/w' + str(train_state_select) + '.txt', W)
np.savetxt(save_folder_name +'/encoded_ '+ str(train_state_select) +'.txt', encoded_st)


# In[ ]:
