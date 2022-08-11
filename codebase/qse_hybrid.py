import numpy as np
import tensorflow as tf
import strawberryfields as sf
from strawberryfields import ops
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from sklearn.metrics import accuracy_score, precision_score, recall_score
import math


from classical_autoencoder import Autoencoder
import params
from state_viz import wigner

def init_weights(modes, layers, active_sd=params.active_sd, passive_sd=params.passive_sd):
    """Initialize a 2D TensorFlow Variable containing normally-distributed
    random weights for an ``N`` mode quantum neural network with ``L`` layers.

    Args:
        modes (int): the number of modes in the quantum neural network
        layers (int): the number of layers in the quantum neural network
        active_sd (float): the standard deviation used when initializing
            the normally-distributed weights for the active parameters
            (displacement, squeezing, and Kerr magnitude)
        passive_sd (float): the standard deviation used when initializing
            the normally-distributed weights for the passive parameters
            (beamsplitter angles and all gate phases)

    Returns:
        tf.Variable[tf.float32]: A TensorFlow Variable of shape
        ``[layers, 2*(max(1, modes-1) + modes**2 + modes)]``, where the Lth
        row represents the layer parameters for the Lth layer.
    """
    # Number of interferometer parameters:
    M = int(modes * (modes - 1)) + max(1, modes - 1)

    # Create the TensorFlow variables
    int1_weights = tf.random.normal(shape=[layers, M], stddev=passive_sd)
    s_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)
    int2_weights = tf.random.normal(shape=[layers, M], stddev=passive_sd)
    dr_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)
    dp_weights = tf.random.normal(shape=[layers, modes], stddev=passive_sd)
    k_weights = tf.random.normal(shape=[layers, modes], stddev=active_sd)

    weights = tf.concat(
        [int1_weights, s_weights, int2_weights, dr_weights, dp_weights, k_weights], axis=1
    )

    weights = tf.Variable(weights)

    return weights

def GenerateTargetState(input_len, f):
    state = np.zeros(input_len)
    state[f] = 1.0
    train = np.array([state])
    return train, state

def layer(params, q, encoded_st):
    """CV quantum neural network layer acting on ``N`` modes.

    Args:
        params (list[float]): list of length ``2*(max(1, N-1) + N**2 + n)`` containing
            the number of parameters for the layer
        q (list[RegRef]): list of Strawberry Fields quantum registers the layer
            is to be applied to
    """
    ops.Dgate(tf.clip_by_value(encoded_st[0][0], clip_value_min = -5, clip_value_max = 5), math.degrees(encoded_st[0][1])) | q[0]

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

def cost(weights, sf_params, eng, qnn, target_state):
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

    difference = tf.reduce_sum(tf.abs(ket - target_state))
    fidelity = tf.abs(tf.reduce_sum(tf.math.conj(ket) * target_state)) ** 2
    return difference, fidelity, ket, tf.math.real(state.trace())

def STM(x_train, target_state):
    autoencoder = Autoencoder(params.latent_dim)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    history = autoencoder.fit(x_train, x_train,
                    epochs=300,
                    validation_data=(x_train, x_train))

    encoded_st = autoencoder.encoder(x_train).numpy()
    #decoded_st = autoencoder.decoder(encoded_st).numpy()
    classical_loss_stat = history.history["loss"]

    np.savetxt(params.save_folder_name +'/encoded_ '+ str(params.train_state_select) +'.txt', encoded_st)


    # initialize engine and program
    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": params.cutoff_dim})
    qnn = sf.Program(params.modes)

    # initialize QNN weights
    weights = init_weights(params.modes, params.Qlayers) # our TensorFlow weights
    num_params = np.prod(weights.shape)   # total number of parameters in our model


    # Create array of Strawberry Fields symbolic gate arguments, matching
    # the size of the weights Variable.
    sf_params = np.arange(num_params).reshape(weights.shape).astype(np.str)
    sf_params = np.array([qnn.params(*i) for i in sf_params])


    # Construct the symbolic Strawberry Fields program by
    # looping and applying layers to the program.
    with qnn.context as q:
        for k in range(params.Qlayers):
            layer(sf_params[k], q, encoded_st)

    opt = tf.keras.optimizers.Adam(learning_rate=params.lr)

    fid_progress = []
    loss_progress = []
    best_fid = 0

    for i in range(params.reps):
        # reset the engine if it has already been executed
        if eng.run_progs:
            eng.reset()

        with tf.GradientTape() as tape:
            loss, fid, ket, trace = cost(weights, sf_params, eng, qnn, target_state)

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
    np.savetxt(params.save_folder_name + '/fidelity' + '.txt', fid_progress)
    np.savetxt(params.save_folder_name + '/loss' + '.txt', fid_progress)
    rho_target = np.outer(target_state, target_state.conj())
    rho_learnt = np.outer(learnt_state, learnt_state.conj())
    X, P, W = wigner(rho_learnt)
    np.savetxt(params.save_folder_name + '/x' + str(params.train_state_select) + '.txt', X)
    np.savetxt(params.save_folder_name + '/p' + str(params.train_state_select) + '.txt', P)
    np.savetxt(params.save_folder_name + '/w' + str(params.train_state_select) + '.txt', W)

def SCTM(x_train, target_state):
    fid_progress = [[], [], [], [], [], [], []]
    loss_progress = [[], [], [], [], [], [], []]


    for j in range(params.sctm_iterations):
        history = autoencoder.fit(x_train, x_train, epochs=params.classical_epochs)

        encoded_st = autoencoder.encoder(np.array([target_state])).numpy()
        #decoded_st = autoencoder.decoder(encoded_st).numpy()

        # initialize engine and program
        eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": params.cutoff_dim})
        qnn = sf.Program(params.modes)

        # initialize QNN weights
        weights = init_weights(params.modes, params.Qlayers) # our TensorFlow weights
        num_params = np.prod(weights.shape)   # total number of parameters in our model

        # Create array of Strawberry Fields symbolic gate arguments, matching
        # the size of the weights Variable.
        sf_params = np.arange(num_params).reshape(weights.shape).astype(np.str)
        sf_params = np.array([qnn.params(*i) for i in sf_params])


        # Construct the symbolic Strawberry Fields program by
        # looping and applying layers to the program.
        with qnn.context as q:
            for k in range(params.Qlayers):
                layer(sf_params[k], q, encoded_st)


        opt = tf.keras.optimizers.Adam(learning_rate=params.lr)

        fidp = []
        lossp = []
        best_fid = 0

        for i in range(params.reps):
            # reset the engine if it has already been executed
            if eng.run_progs:
                eng.reset()

            with tf.GradientTape() as tape:
                loss, fid, ket, trace = cost(weights, sf_params, eng, qnn, target_state)

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

def main():
    x_train, target_state = GenerateTargetState(params.input_len, params.train_state_select)
    print('The target state for the training is chosen to be ' + str(target_state))
    print('The selected method is: ' + params.method_select)
    if (params.method_select == 'STM'):
        STM(x_train, target_state)
    elif (params.method_select == 'SCTM'):
        SCTM(x_train, target_state)
    else:
        print('Invalid method selection in terminal. ')
