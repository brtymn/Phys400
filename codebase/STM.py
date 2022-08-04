#!/usr/bin/env python
# coding: utf-8

# # Split Training Model For The Hybrid Autoencoder

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




autoencoder = Autoencoder(latent_dim)


x_train, target_state = GenerateTargetState(input_len, train_state_select)
print('The target state for the training is chosen to be ' + str(target_state))


# ### Compile the classical autoencoder.

# In[6]:


autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())


# ### Train the classical autoencoder and obtain the latent space.

# In[7]:


history = autoencoder.fit(x_train, x_train,
                epochs=300,
                validation_data=(x_train, x_train))


# In[8]:


encoded_st = autoencoder.encoder(x_train).numpy()
decoded_st = autoencoder.decoder(encoded_st).numpy()


# In[9]:


plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()


# In[10]:


print(encoded_st)
np.savetxt(save_folder_name +'/encoded_ '+ str(train_state_select) +'.txt', encoded_st)


# In[11]:


import strawberryfields as sf
from strawberryfields import ops
sf.about()


# In[12]:


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


# ### Initialization of layer weights.

# In[13]:



# In[14]:


# initialize engine and program
eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff_dim})
qnn = sf.Program(modes)

# initialize QNN weights
weights = init_weights(modes, Qlayers) # our TensorFlow weights
num_params = np.prod(weights.shape)   # total number of parameters in our model


# In[15]:


# Create array of Strawberry Fields symbolic gate arguments, matching
# the size of the weights Variable.
sf_params = np.arange(num_params).reshape(weights.shape).astype(np.str)
sf_params = np.array([qnn.params(*i) for i in sf_params])


# Construct the symbolic Strawberry Fields program by
# looping and applying layers to the program.
with qnn.context as q:
    for k in range(Qlayers):
        layer(sf_params[k], q)


# ### Definition of the quantum cost function.

# In[16]:


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


# ### Optimizer definition.

# In[17]:


opt = tf.keras.optimizers.Adam(learning_rate=lr)


# ### Train the quantum decoder.

# In[18]:


fid_progress = []
loss_progress = []
best_fid = 0

for i in range(reps):
    # reset the engine if it has already been executed
    if eng.run_progs:
        eng.reset()

    with tf.GradientTape() as tape:
        loss, fid, ket, trace = cost(weights)

    # Stores fidelity at each step
    fid_progress.append(fid.numpy())

    loss_progress.append(loss)

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


# ### Produce fidelity plot.

# In[19]:


from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

fig, ax = plt.subplots(figsize=(12, 6), dpi = 100)

# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(16)

plt.plot(fid_progress, linewidth = 3)
plt.ylabel("Fidelity", fontsize = '16')
ax.yaxis.set_ticks(np.arange(0, 1, 0.4))
plt.xlabel("Epoch", fontsize = '16')
ax.xaxis.set_ticks(np.arange(0, reps, 200))
plt.title('STM Fidelity', fontsize = '20')


plt.savefig('STM_fidelity.png')


# ### Produce loss plot.

# In[20]:


fig, ax = plt.subplots(figsize=(12, 12), dpi = 100)

# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(16)

plt.plot(loss_progress, linewidth = 3)
plt.ylabel("Training Loss", fontsize = '20')
ax.yaxis.set_ticks(np.arange(0, 1, 0.4))
plt.xlabel("Epoch", fontsize = '20')
ax.xaxis.set_ticks(np.arange(0, reps, 200))
plt.title('STM Training Loss', fontsize = '24')


plt.savefig('STM_loss.png')


# ### Definition of the function that plots Wigner functions.

# In[21]:


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

# In[22]:


rho_target = np.outer(target_state, target_state.conj())
rho_learnt = np.outer(learnt_state, learnt_state.conj())


# ### Plot the target state as a Wigner function.

# In[23]:


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
X, P, W = wigner(rho_target)
ax.plot_surface(X, P, W, cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)
ax.contour(X, P, W, 10, cmap="RdYlGn", linestyles="solid", offset=-0.17)
ax.set_axis_off()
fig.show()


# ### Plot the learnt state as a Wigner function.

# In[24]:


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
X, P, W = wigner(rho_learnt)
ax.plot_surface(X, P, W, cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)
ax.contour(X, P, W, 10, cmap="RdYlGn", linestyles="solid", offset=-0.17)
ax.set_axis_off()
fig.show()


# In[25]:


np.savetxt(save_folder_name + '/x' + str(train_state_select) + '.txt', X)
np.savetxt(save_folder_name + '/p' + str(train_state_select) + '.txt', P)
np.savetxt(save_folder_name + '/w' + str(train_state_select) + '.txt', W)


# In[ ]:
