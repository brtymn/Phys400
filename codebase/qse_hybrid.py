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


def init_weights(modes, layers, active_sd=0.0001, passive_sd=0.1):
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
