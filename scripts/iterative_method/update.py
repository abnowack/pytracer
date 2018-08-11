"""
Tasks:
    [x] - Fake estimate plot setup
    [ ] - Load measurement data
    [ ] - Get discretized transmission estimate
    [ ] - Reconstruct mu_f estimate given est.p
    [ ] - Create p recon algorithm
"""

import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple

from scripts.assemblies import shielded_assembly
import pytracer.transmission as transmission
import pytracer.fission as fission
import pytracer.neutron_chain as neutron_chain
import pytracer.geometry as geo

Data = namedtuple('Data', ['trans', 'f_1', 'f_2'])
Estimate = namedtuple('Estimate', ['extent', 'mu', 'mu_f', 'p'])
Image = namedtuple('Image', ['data', 'x_min', 'x_max', 'y_min', 'y_max'])


def load_data(path=r'scripts\data'):
    assembly_solids = shielded_assembly()
    assembly_flat = geo.flatten(assembly_solids)

    radians = np.linspace(0, np.pi, 100)
    a = np.zeros
    arc_radians = np.linspace(-np.pi / 8, np.pi / 8, 100)
    source, detector_points, extent = geo.fan_beam_paths(60, arc_radians, radians, extent=True)
    source = source[0, :, :]

    neutron_chain.generate_p_matrix(neutron_chain.nu_pu239_induced,
                                    path + r'\nudist_matrix_pu239.npz')

    with np.load(path + r'\nudist_matrix_pu239.npz') as data:
        p_range = data['p_range']
        matrix = data['matrix']

    single_probs = fission.scan(source, detector_points, detector_points, assembly_flat, 1, matrix, p_range)
    np.save(path + r'\single_probs_vary_p', single_probs)

    double_probs = fission.scan(source, detector_points, detector_points, assembly_flat, 2, matrix, p_range)
    np.save(path + r'\double_probs_vary_p', double_probs)


def initial_estimate():
    xs = np.linspace(-11, 11, 200)
    ys = np.linspace(-6, 6, 200)

    assembly = shielded_assembly(fission=True)
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


def update(data, estimate):
    if estimate['mu'] is None:
        pass
        # estimate transmission

    estimate['mu_f'] = estimate_mu_f(data, estimate['mu'], estimate['p'])
    estimate['p'] = estimate_p(data, estimate['mu'], estimate['mu_f'])

    return estimate


if __name__ == '__main__':
    # data = Data()
    estimate = initial_estimate()

    plt.figure()
    plt.imshow(estimate.mu, extent=estimate.extent)

    plt.figure()
    plt.imshow(estimate.mu_f, extent=estimate.extent)

    plt.figure()
    plt.imshow(estimate.p, extent=estimate.extent)

    plt.show()
