import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
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

def WignerFUncPlot():
    rho_target = np.outer(target_state, target_state.conj())
    rho_learnt = np.outer(learnt_state, learnt_state.conj())
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    X, P, W = wigner(rho_target)
    ax.plot_surface(X, P, W, cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)
    ax.contour(X, P, W, 10, cmap="RdYlGn", linestyles="solid", offset=-0.17)
    ax.set_axis_off()
    fig.show()


def FidelityPot():

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


def LossPlot():
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
