"""
Tasks:
    [x] - Fake estimate plot setup
    [x] - Load measurement data
    [ ] - Get discretized transmission estimate
    [ ] - Reconstruct mu_f estimate given est.p
    [ ] - Create p recon algorithm
"""

from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pathlib

import pytracer.fission as fission
import pytracer.geometry as geo
import pytracer.neutron_chain as neutron_chain
import pytracer.transmission as transmission
from scripts.assemblies import shielded_assembly

Data = namedtuple('Data', ['extent', 'trans', 'f_1', 'f_2'])
Estimate = namedtuple('Estimate', ['extent', 'mu', 'mu_f', 'p'])


def load_data(directory=r'scripts\data', data_filename='probs'):
    path = pathlib.Path.cwd() / directory

    data_probs = np.load(path / (data_filename + '.npz'))
    return Data(data_probs['extent'], data_probs['trans'], data_probs['f_1'], data_probs['f_2'])


def create_data(directory=r'scripts\data', p_matrix_name='nudist_matrix_pu239', recreate_p_matrix=True,
                data_filename='probs'):

    path = pathlib.Path.cwd() / directory

    print('Generating Fission Probability Distributions... ', flush=True)
    p_matrix_path = path / (p_matrix_name + '.npz')
    if not recreate_p_matrix:
        matrix_loaded = np.load(p_matrix_path)
        matrix = matrix_loaded['matrix']
        p_range = matrix_loaded['p_range']
    else:
        # TODO: Add option to change nuclear data
        matrix, p_range = neutron_chain.generate_p_matrix(neutron_chain.nu_pu239_induced, max_n=20, p_range=20)
        np.savez(p_matrix_path, matrix=matrix, p_range=p_range)
    print('Done', flush=True)

    print('Generating Geometry... ', flush=True)
    assembly_solids = shielded_assembly()
    assembly_flat = geo.flatten(assembly_solids)

    radians = np.linspace(0, np.pi, 100)
    arc_radians = np.linspace(-np.pi / 8, np.pi / 8, 100)
    source, detector_points, extent = geo.fan_beam_paths(60, arc_radians, radians, extent=True)
    print('Done', flush=True)

    print('Generating Transmission Scan...', flush=True)
    trans_probs = transmission.scan(assembly_flat, source, detector_points)
    print('Done', flush=True)

    print('Generating Single Neutron Scan... ', flush=True)
    single_probs = fission.scan(source[0, :, :], detector_points, detector_points, assembly_flat, 1, matrix, p_range)
    print('Done', flush=True)

    print('Generating Double Neutron Scan... ', flush=True)
    double_probs = fission.scan(source[0, :, :], detector_points, detector_points, assembly_flat, 2, matrix, p_range)
    print('Done', flush=True)

    data_path = path / (data_filename + '.npz')
    np.savez(data_path, extent=extent, trans=trans_probs, f_1=single_probs, f_2=double_probs)

    return Data(extent, trans_probs, single_probs, double_probs)


def initial_estimate():
    xs = np.linspace(-11, 11, 200)
    ys = np.linspace(-6, 6, 200)

    assembly = shielded_assembly()
    assembly_flat = geo.flatten(assembly)

    mu_image, extent = transmission.absorbance_image(xs, ys, assembly_flat)
    mu_f_image, extent = fission.fissionval_image(xs, ys, assembly_flat)
    p_image, extent = neutron_chain.p_image(xs, ys, assembly_flat)

    est = Estimate(extent=extent, mu=mu_image, mu_f=mu_f_image, p=p_image)

    return est


def estimate_mu_f(data, mu, p):
    pass


def estimate_p(data, mu, mu_f):
    pass


def update_estimate(data, estimate):
    if estimate['mu'] is None:
        pass
        # estimate transmission

    estimate['mu_f'] = estimate_mu_f(data, estimate['mu'], estimate['p'])
    estimate['p'] = estimate_p(data, estimate['mu'], estimate['mu_f'])

    return estimate


def display_estimate(estimate):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, (mu_ax, mu_f_ax, p_ax) = plt.subplots(ncols=1, nrows=3, sharex=True, constrained_layout=True, figsize=(5, 6))

    mu_im = mu_ax.imshow(estimate.mu, extent=estimate.extent)
    mu_divider = make_axes_locatable(mu_ax)
    mu_ax_cb = mu_divider.new_horizontal(size="5%", pad=0.05)
    mu_fig = mu_ax.get_figure()
    mu_fig.add_axes(mu_ax_cb)
    mu_cb = plt.colorbar(mu_im, cax=mu_ax_cb)
    mu_cb.ax.set_ylabel(r'$\mu$', rotation=0, labelpad=10, size=15)

    mu_f_im = mu_f_ax.imshow(estimate.mu_f, extent=estimate.extent)
    mu_f_divider = make_axes_locatable(mu_f_ax)
    mu_f_ax_cb = mu_f_divider.new_horizontal(size="5%", pad=0.05)
    mu_f_fig = mu_f_ax.get_figure()
    mu_f_fig.add_axes(mu_f_ax_cb)
    mu_f_cb = plt.colorbar(mu_f_im, cax=mu_f_ax_cb)
    mu_f_cb.ax.set_ylabel(r'$\frac{\mu_f}{\mu}$', rotation=0, labelpad=10, size=15)

    p_im = p_ax.imshow(estimate.p, extent=estimate.extent)
    p_divider = make_axes_locatable(p_ax)
    p_ax_cb = p_divider.new_horizontal(size="5%", pad=0.05)
    p_fig = p_ax.get_figure()
    p_fig.add_axes(p_ax_cb)
    p_cb = plt.colorbar(p_im, cax=p_ax_cb)
    p_cb.ax.set_ylabel(r'$p$', rotation=0, labelpad=10, size=15)

    mu_f_ax.set_ylabel('Y (cm)')
    p_ax.set_xlabel('X (cm)')
    fig.suptitle('Estimate Results')


def display_data(data):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, (trans_ax, f_1_ax, f_2_ax) = plt.subplots(ncols=1, nrows=3, sharex=True, constrained_layout=True, figsize=(6, 4))
    trans_im = trans_ax.imshow(data.trans, extent=data.extent)
    trans_divider = make_axes_locatable(trans_ax)
    trans_ax_cb = trans_divider.new_horizontal(size="3%", pad=0.05)
    trans_fig = trans_ax.get_figure()
    trans_fig.add_axes(trans_ax_cb)
    trans_cb = plt.colorbar(trans_im, cax=trans_ax_cb)
    trans_cb.ax.set_ylabel(r'$P_{trans}$', rotation=0, labelpad=25, size=15)

    f_1_im = f_1_ax.imshow(data.f_1.T, extent=data.extent)
    f_1_divider = make_axes_locatable(f_1_ax)
    f_1_ax_cb = f_1_divider.new_horizontal(size="3%", pad=0.05)
    f_1_fig = f_1_ax.get_figure()
    f_1_fig.add_axes(f_1_ax_cb)
    f_1_cb = plt.colorbar(f_1_im, cax=f_1_ax_cb)
    f_1_cb.ax.set_ylabel(r'$P_{f1}$', rotation=0, labelpad=15, size=15)

    f_2_im = f_2_ax.imshow(data.f_2.T, extent=data.extent)
    f_2_divider = make_axes_locatable(f_2_ax)
    f_2_ax_cb = f_2_divider.new_horizontal(size="3%", pad=0.05)
    f_2_fig = f_2_ax.get_figure()
    f_2_fig.add_axes(f_2_ax_cb)
    f_2_cb = plt.colorbar(f_2_im, cax=f_2_ax_cb)
    f_2_cb.ax.set_ylabel(r'$P_{f2}$', rotation=0, labelpad=13, size=15)

    f_1_ax.set_ylabel(r'Neutron Angle $\phi$ (rad)')
    f_2_ax.set_xlabel(r'Detector Angle $\theta$ (rad)')
    fig.suptitle('Data Sinograms')
    plt.subplots_adjust(top=0.92, bottom=0.125, left=0.100, right=0.9, hspace=0.2, wspace=0.2)


if __name__ == '__main__':
    # data = create_data(recreate_p_matrix=False)
    data = load_data()
    estimate = initial_estimate()

    print(data.extent)
    print(estimate.extent)

    display_data(data)
    display_estimate(estimate)

    plt.show()
