import tensorflow as tf
import numpy as np
import strawberryfields as sf
from classical_autoencoder import Autoencoder

'''
Split Traning Model function call.
'''
def STM(autoencoder, x_train):
    history = autoencoder.fit(x_train, x_train,
                    epochs=300,
                    validation_data=(x_train, x_train))


    encoded_st = autoencoder.encoder(x_train).numpy()
    #decoded_st = autoencoder.decoder(encoded_st).numpy()
    classical_loss_stat = history.history["loss"]

    np.savetxt(save_folder_name +'/encoded_ '+ str(train_state_select) +'.txt', encoded_st)


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
        np.savetxt(save_folder_name + '/fidelity' + '.txt', fid_progress)
        np.savetxt(save_folder_name + '/loss' + '.txt', fid_progress)
        X, P, W = wigner(rho_learnt)
        np.savetxt(save_folder_name + 'STM//x' + str(train_state_select) + '.txt', X)
        np.savetxt(save_folder_name + 'STM//p' + str(train_state_select) + '.txt', P)
        np.savetxt(save_folder_name + 'STM/w' + str(train_state_select) + '.txt', W)
