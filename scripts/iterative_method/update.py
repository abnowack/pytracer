"""
Tasks:
    [x] - Fake estimate plot setup
    [x] - Load measurement data
    [ ] - Create response matrices
    [ ] - Move data / image / simulation routines into other file, separate reconstruction into this file
    [ ] - Get forward projection working again
    [ ] - Have responses use an image estimate of p
    [ ] - Reconstruct mu_f estimate given est.p
    [ ] - Create p recon algorithm
"""

import matplotlib.pyplot as plt
import numpy as np

import pytracer.fission as fission
import pytracer.transmission as transmission
import pytracer.geometry as geo
import pytracer.neutron_chain as neutron_chain
import pytracer.measurement as measure
import pytracer.response as response
from scripts.assemblies import shielded_assembly
from scripts.utils import Data, Estimate


def initial_estimate():
    xs = np.linspace(-11, 11, 200)
    ys = np.linspace(-6, 6, 200)

    assembly = shielded_assembly()
    assembly_flat = geo.flatten(assembly)

    # truth
    mu_image, extent = transmission.absorbance_image(xs, ys, assembly_flat)
    mu_f_image, extent = fission.fissionval_image(xs, ys, assembly_flat)
    p_image, extent = neutron_chain.p_image(xs, ys, assembly_flat)

    #

    return Estimate(extent=extent, mu=mu_image, mu_f=mu_f_image, p=p_image)


def estimate_mu(data, estimate):
    pass


def estimate_mu_f(data, estimate):
    pass


def estimate_p(data, estimate):
    pass


def update_estimate(data, estimate):

    estimate['mu'] = estimate_mu(data, estimate)
    estimate['mu_f'] = estimate_mu_f(data, estimate)
    estimate['p'] = estimate_p(data, estimate)

    return estimate


if __name__ == '__main__':
    from scripts.utils import display_data, display_estimate, load_data

    # deriv_data = create_data(load_p_matrix=True, deriv=True, data_filename='probs_deriv')
    # data = create_data(load_p_matrix=True, data_filename='probs')
    deriv_data = load_data(data_filename='probs_deriv')
    data = load_data(data_filename='probs')

    diff_data = Data(deriv_data.extent, deriv_data.trans - data.trans, deriv_data.f_1 - data.f_1,
                     deriv_data.f_2 - data.f_2)

    display_data(diff_data)
    display_data(data)

    estimate = initial_estimate()
    display_estimate(estimate)

    plt.show()
