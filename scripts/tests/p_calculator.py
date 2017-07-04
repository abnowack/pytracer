"""
Fix the p_range size issue with the matrix,
"""

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker
import numpy as np
import mpmath as mp
from collections import namedtuple
import os

nu_u235_induced = \
    np.array([0.0237898, 0.1555525, 0.3216515, 0.3150433, 0.1444732, 0.0356013, 0.0034339, 0.0004546])
nu_u238_induced = \
    np.array([0.0798571, 0.2467090, 0.3538440, 0.2356090, 0.0772832, 0.0104521, 0.0006964])
nu_pu239_induced = \
    np.array([0.0084842, 0.079003, 0.2536175, 0.328987, 0.2328111, 0.0800161, 0.0155581, 0.001176, 0.0003469])
nu_pu240_induced = \
    np.array([0.0631852, 0.2319644, 0.3333230, 0.2528207, 0.0986461, 0.0180199, 0.0020406])
nu_u235_spontaneous = \
    np.array([0.0481677, 0.2485215, 0.4253044, 0.2284094, 0.0423438, 0.0072533])
nu_pu240_spontaneous = \
    np.array([0.0631852, 0.2319644, 0.3333230, 0.2528207, 0.0986461, 0.0180199, 0.0020406])


def critical_p(nu_dist):
    crit_value = 1 / np.sum(nu_dist * np.arange(len(nu_dist)))
    return crit_value


def calc_h0(nu, p):
    coeffs = p * nu
    coeffs[1] -= 1

    poly = np.poly1d(coeffs[::-1])
    roots = poly.r
    real_roots = roots[np.isreal(roots)].real

    r = np.sort(real_roots)
    r = r[r >= 0]
    return r[0]


def calc_term_noonan(p, nudist, n, dps=None):
    if dps:
        mp.dps = dps

    coeffs = nudist[::-1].tolist()

    def calc_term(z, nudist, p, n):
        ans = mp.polyval(coeffs, z)
        ans = z - p * ans
        ans = (1.0 - p) / ans
        return mp.power(ans, n)

    ans2 = mp.quad(lambda z: calc_term(z, nudist, p, n), [1, mp.j, -1, -mp.j, 1])
    ans2 /= 2 * mp.pi * n
    return ans2.imag


def ndist_noonan(p, nudist, max_n, dps=None):
    ns = np.zeros(max_n + 1)
    if p <= 0.0:
        ns[0] = 1
        return ns
    ns[0] = calc_h0(nudist, p)

    for n in range(1, max_n + 1):
        ns[n] = calc_term_noonan(p, nudist, n, dps)
    return ns


def generate_p_matrix(nu_dist, name, max_n=100, p_range=100, reload=False):
    if type(p_range) is int:
        crit_value = critical_p(nu_dist)
        p_range = np.linspace(0, crit_value, p_range)

    # is matrix already present?
    matrix_name = os.path.join('data', 'pmatrix_' + name + '.npy')
    if not reload:
        try:
            matrix = np.load(matrix_name)
            if np.array_equal(matrix[0, :len(p_range)], p_range):
                return matrix
        except FileNotFoundError:
            pass

    n_p = len(p_range)
    matrix = np.zeros((n_p + 1, max_n + 1))
    matrix[0, :len(p_range)] = p_range

    for i in range(1, n_p + 1):
        p = p_range[i - 1]
        matrix[i] = ndist_noonan(p, nu_dist, max_n)

    np.save(matrix_name, matrix)

    return matrix


def interpolate_p(matrix, p_value, p_range, method='linear'):
    # p_range = matrix[0]  # TODO FIX

    if method == 'linear':
        low_index = int(np.digitize(p_value, p_range))
        high_index = low_index + 1
        low_value = p_range[low_index]
        high_value = p_range[high_index]

        delta = high_value - low_value
        left_delta = p_value - low_value
        right_delta = high_value - p_value
        left_frac = left_delta / delta
        right_frac = right_delta / delta

        return left_frac * matrix[low_index + 1] + right_frac * matrix[high_index + 1]


if __name__ == '__main__':

    p_range = np.linspace(0, 0.2, 10)
    p_matrix = generate_p_matrix(nu_pu239_induced, 'pu239', max_n=20, p_range=p_range)

    print(len(p_range), 21, p_matrix[1:].shape)


    def log_tick_formatter(val, pos=None):
        return "{:.2e}".format(10 ** val)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs, ys = np.meshgrid(p_range[1:], list(range(21)))
    # ax.plot_surface(xs, ys, p_matrix[1:].T)

    minz = np.log10(min(p_matrix[1:, :][p_matrix[1:, :] > 0.0]))
    print(minz)

    surf = ax.plot_surface(xs, ys, np.log10(p_matrix[2:].T), cmap=cm.coolwarm,
                           norm=colors.Normalize(vmin=minz, vmax=1.0))
    ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    exp_prange = np.linspace(0, 0.2, 20)
    xs, ys = np.meshgrid(exp_prange[1:], list(range(21)))
    exp_pmatrix = np.zeros((len(exp_prange), 21))

    for i in range(len(exp_prange)):
        exp_pmatrix[i] = interpolate_p(p_matrix, exp_prange[i], p_range)

    minz = np.log10(min(exp_pmatrix[1:, :][exp_pmatrix[1:, :] > 0.0]))
    print(minz)

    surf = ax2.plot_surface(xs, ys, np.log10(exp_pmatrix[2:].T), cmap=cm.coolwarm,
                            norm=colors.Normalize(vmin=minz, vmax=1.0))
    ax2.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))

    plt.show()
