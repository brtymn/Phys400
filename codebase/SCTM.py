import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from sklearn.metrics import accuracy_score, precision_score, recall_score

import matplotlib.pyplot as plt
import numpy as np
import math



fid_progress = [[], [], [], [], [], [], []]
loss_progress = [[], [], [], [], [], [], []]


for j in range(sctm_iterations):
    history = autoencoder.fit(x_train, x_train,epochs=classical_epochs)

    encoded_st = autoencoder.encoder(np.array([target_state])).numpy()
    #decoded_st = autoencoder.decoder(encoded_st).numpy()

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


# ### Plot the target state as a Wigner function.

# In[13]:


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
X, P, W = wigner(rho_target)
ax.plot_surface(X, P, W, cmap="Spectral", lw=0.5, rstride=1, cstride=1)
ax.contour(X, P, W, 10, cmap="Spectral", linestyles="solid", offset=-0.17)
ax.set_axis_off()
fig.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
X, P, W = wigner(rho_learnt)
ax.plot_surface(X, P, W, cmap="Spectral", lw=0.5, rstride=1, cstride=1)
ax.contour(X, P, W, 10, cmap="Spectral", linestyles="solid", offset=-0.17)
ax.set_axis_off()
fig.show()




np.savetxt(save_folder_name + '/x' + str(train_state_select) + '.txt', X)
np.savetxt(save_folder_name + '/p' + str(train_state_select) + '.txt', P)
np.savetxt(save_folder_name + '/w' + str(train_state_select) + '.txt', W)
np.savetxt(save_folder_name +'/encoded_ '+ str(train_state_select) +'.txt', encoded_st)
