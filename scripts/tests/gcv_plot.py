"""
Create response matrices for the single and double fission responses over a grid, for the basic assembly geometry.
"""

import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import numpy as np
from scripts.assemblies import shielded_assembly
import pytracer.geometry as geo
import pytracer.transmission as transmission
import pytracer.algorithms as algorithms
import pytracer.fission as fission


def nice_double_plot(data1, data2, extent, title1='', title2='', xlabel='', ylabel=''):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    print(data1.min(), data1.max())
    im1 = ax1.imshow(data1, interpolation='none', extent=extent, cmap='viridis')
    ax1.set_title(title1)
    # vmin1, vmax1 = im1.get_clim()
    # print(vmin1, vmax1)

    im2 = ax2.imshow(data2, interpolation='none', extent=extent, cmap='viridis')
    ax2.set_title(title2)
    # vmin2, vmax2 = im2.get_clim()
    # im2.set_clim(vmin1, vmax1)
    # print(vmin2, vmax2)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.yaxis.labelpad = 40
    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    plt.subplots_adjust(right=0.98, top=0.95, bottom=0.07, left=0.12, hspace=0.05, wspace=0.20)


def gvc_svd(d, G, alphas):
    gcvs = np.zeros(len(alphas))

    U, s, V = np.linalg.svd(G, full_matrices=True)
    F = np.zeros((np.size(U, 0), np.size(U, 0)))
    filter = np.zeros(np.size(U, 0))

    for i, alpha in enumerate(alphas):
        filter[:len(s)] = s ** 2 / (s ** 2 + alpha ** 2)
        # np.fill_diagonal(F, filter)
        H = (U * filter) @ U.T
        num = np.size(G, 0) * np.linalg.norm(H @ d - d) ** 2
        denom = np.trace(H - np.identity(np.size(G, 0))) ** 2
        gcvs[i] = num / denom
        print(i, alpha, gcvs[i])

    return gcvs


if __name__ == "__main__":
    response_single = np.load(r'data\fission_response_single.npy')
    response_double = np.load(r'data\fission_response_double.npy')
    single_probs = np.load(r'data\single_probs.npy')
    double_probs = np.load(r'data\double_probs.npy')

    d1 = single_probs.reshape((-1))
    G1 = response_single.reshape((response_single.shape[0], -1)).T
    d2 = single_probs.reshape((-1))
    G2 = response_single.reshape((response_double.shape[0], -1)).T

    #### Single Neutron
    # alphas = np.logspace(-5, -1, 100)
    # norms, residuals = algorithms.trace_lcurve(d1, G1, alphas)
    # plt.loglog(residuals, norms)
    #
    # plt.figure()
    # c_alphas, curve = algorithms.lcurve_curvature(alphas, norms, residuals)
    # plt.semilogx(c_alphas, curve)

    # m1_alpha = algorithms.solve_tikhonov(d1, G1, 0.007)
    # m1_alpha = m1_alpha.reshape((30, 50))

    # plt.figure()
    # plt.imshow(m1_alpha, interpolation='none', vmax=1.0, vmin=0)
    # plt.colorbar()

    #### Double Neutron
    # alphas = np.logspace(-5, -1, 100)
    # norms, residuals = algorithms.trace_lcurve(d2, G2, alphas)
    # plt.loglog(residuals, norms)
    #
    # plt.figure()
    # c_alphas, curve = algorithms.lcurve_curvature(alphas, norms, residuals)
    # plt.semilogx(c_alphas, curve)

    # m2_alpha = algorithms.solve_tikhonov(d2, G2, 0.007)
    # m2_alpha = m2_alpha.reshape((30, 50))

    # plt.figure()
    # plt.imshow(m2_alpha, interpolation='none', vmax=1.0, vmin=0)
    # plt.colorbar()

    #### Single + Double Neutron
    d = np.concatenate((d1, d2)) * 0.2
    G = np.concatenate((G1, G2))

    alphas = np.logspace(-6, 0, 50)
    norms, residuals = algorithms.trace_lcurve(d, G, alphas)

    plt.figure()
    c_alphas, curve = algorithms.lcurve_curvature(alphas, norms, residuals)
    plt.semilogx(c_alphas, curve)

    best_alpha_arg = np.argmax(curve)
    best_alpha = c_alphas[best_alpha_arg]
    print(best_alpha)

    norms[0] *= 10
    plt.figure()
    plt.loglog(residuals, norms, 'k', lw=2)
    print(alphas)
    print(norms)
    plt.xlabel(r'$\log{||A X - Y||_2^2}$', size=20)
    plt.ylabel(r'$\log{||X||_2^2}$', size=20)
    plt.title(r'L-Curve Criterion with Optimal $\alpha$ = {:.3f}'.format(best_alpha), size=15)
    plt.scatter(residuals[best_alpha_arg], norms[best_alpha_arg], c='red', s=100, edgecolors='none')
    plt.tight_layout()

    m_alpha = algorithms.solve_tikhonov(d, G, 0.01)
    m_alpha = m_alpha.reshape((30, 50))

    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    im = plt.imshow(m_alpha, interpolation='none', vmax=0.2, vmin=0, extent=[-25. / 2, 25. / 2, -15. / 2, 15. / 2],
                    cmap='viridis')
    plt.title('Single & Double Reconstruction', size=20)
    plt.xlabel('X (cm)', size=18)
    plt.ylabel('Y (cm)', size=18)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(r'$\mu_f / \mu_{total}$', size=20, labelpad=15)

    plt.show()
