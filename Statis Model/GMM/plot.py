import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import matplotlib as mpl
import pandas as pd


def plot_1D(gmm, x, col):
    plt.hist(x, density=True)
    x = np.linspace(x.min(), x.max(), 100, endpoint=False)
    ys = np.zeros_like(x)

    j = 0
    for p in gmm.p_arr:
        y = p * st.multivariate_normal.pdf(x, mean=gmm.mu_arr[j], cov=gmm.Sigma_arr[j])
        plt.plot(x, y)
        ys += y
        j += 1

    plt.xlabel(col)
    plt.plot(x, ys)
    plt.show()


def make_ellipses(gmm, ax):
    colors = ['lightcoral', 'mediumpurple']

    for n, color in enumerate(colors):
        convariances = gmm.Sigma_arr[n]
        v, w = np.linalg.eigh(convariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi
        print(f'w = {w}, v = {v}, angle={angle}')
        v = 3. * np.sqrt(2.) * np.sqrt(v)
        mean = gmm.mu_arr[n]
        mean = mean.reshape(2, 1)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')


def plot_2D(gmm, x, col, label):
    h = plt.subplot(111, aspect='equal')
    make_ellipses(gmm, h)

    plt.scatter(x[:, 0], x[:, 1], c=label['Species'], marker='x')
    plt.xlim(-3, 9)
    plt.ylim(-3, 9)
    plt.xlabel(col[0])
    plt.ylabel(col[1])
    plt.show()