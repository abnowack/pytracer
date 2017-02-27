"""
Create response matrices for the single and double fission responses over a grid, for the basic assembly geometry.
"""

import sys
import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    assembly_solids = shielded_assembly()
    assembly_flat = geo.flatten(assembly_solids)

    nu_dist = np.array([0.0481677, 0.2485215, 0.4253044, 0.2284094, 0.0423438, 0.0072533], dtype=np.double)
    nu_dist /= np.sum(nu_dist)

    plt.figure()
    geo.draw(assembly_solids)

    radians = np.linspace(0, np.pi, 100)
    arc_radians = np.linspace(-np.pi / 8, np.pi / 8, 100)
    source, detector_points, extent = geo.fan_beam_paths(60, arc_radians, radians, extent=True)
    source = source[0, :, :]

    # grid = geo.Grid(width=25, height=15, num_x=50, num_y=30)
    grid = geo.Grid(width=25, height=15, num_x=25, num_y=15)
    grid.draw()

    cell_i = 520 / 4 + 5
    grid_points = grid.cell(cell_i)
    plt.fill(grid_points[:, 0], grid_points[:, 1], color='blue', zorder=12)
    plt.xlabel('X (cm)', size=18)
    plt.ylabel('Y (cm)', size=18)
    ax = plt.gca()
    ax.set_aspect('equal')
    # plt.scatter(source[0, 0], source[0, 1])

    # response_single = fission.grid_response(source, detector_points, detector_points, grid, assembly_flat, 1, nu_dist)
    # np.save(r'data\fission_response_single', response_single)
    # response_single = np.load(r'data\fission_response_single.npy')

    # response_double = fission.grid_response(source, detector_points, detector_points, grid, assembly_flat, 2, nu_dist)
    # np.save(r'data\fission_response_double', response_double)
    # response_double = np.load(r'data\fission_response_double.npy')

    # single_probs = fission.scan(source, detector_points, detector_points, assembly_flat, 1, nu_dist)
    # np.save(r'data\single_probs', single_probs)
    # single_probs = np.load(r'data\single_probs.npy')

    # double_probs = fission.scan(source, detector_points, detector_points, assembly_flat, 2, nu_dist)
    # np.save(r'data\double_probs', double_probs)
    # double_probs = np.load(r'data\double_probs.npy')


    # nice_double_plot(response_single[cell_i].T, response_double[cell_i].T, extent, 'Single Fission Response',
    #                  'Double Fission Response', 'Detector Orientation Angle', 'Source Neutron Direction Angle')
    #
    # nice_double_plot(single_probs.T, double_probs.T, extent, 'Single Neutron Probability',
    #                  'Double Neutron Probability', 'Detector Orientation Angle', 'Source Neutron Direction Angle')

    # alphas = np.linspace(1e-7, 0.3, 500)
    # norm, residual = algorithms.trace_lcurve(single_probs, response_single, alphas)
    # plt.figure()
    # plt.loglog(residual, norm, marker='o')
    # plt.title('LCurve Single Fission')

    # plt.figure()
    # curv = algorithms.lcurve_curvature(np.log(residual), np.log(norm))
    # max_curv = np.argmax(curv)
    # max_alpha = alphas[max_curv + 1]
    # plt.plot(alphas[1:-1], curv)
    # plt.title('Max Single Fission LCurve 2nd Deriv @ alpha = ' + str(max_alpha))

    # recon_single = algorithms.solve_tikhonov(single_probs.reshape((-1)), response_single.reshape((response_single.shape[0], -1)).T, alpha=0.0168)
    # recon_single = recon_single.reshape(grid.num_y, grid.num_x)
    # plt.figure(figsize=(6, 4))
    # plt.imshow(recon_single, interpolation='none', extent=[-25./2, 25./2, -15./2, 15./2], cmap='viridis', vmin=0, vmax=1.0)
    # plt.title('Single Neutron Reconstruction')
    # cbar = plt.colorbar()
    # cbar.ax.get_yaxis().labelpad = 20
    # cbar.ax.set_ylabel(r'$\mu_{fission} / \mu_{total}$', rotation=270, size=17)
    # plt.tight_layout()

    # alphas = np.logspace(-4, 0, 50)
    # gcv_values = np.zeros(alphas.shape)
    # for i, alpha in enumerate(alphas):
    #     gcv_values[i] = algorithms.generalized_cross_validation(single_probs.reshape((-1)), response_single.reshape((response_single.shape[0], -1)).T, alpha)
    #     print(i, alpha, gcv_values[i])

    # gcv = algorithms.generalized_cross_validation(single_probs.reshape((-1)), response_single.reshape((response_single.shape[0], -1)).T, alphas)
    # plt.loglog(alphas, gcv_values)

    # alphas = np.linspace(1e-7, 0.1, 500)
    # norm, residual = algorithms.trace_lcurve(double_probs, response_double, alphas)
    # plt.figure()
    # plt.loglog(residual, norm, marker='o')
    # plt.title('LCurve Double Fission')

    # plt.figure()
    # curv = algorithms.lcurve_curvature(np.log(residual), np.log(norm))
    # max_curv = np.argmax(curv)
    # max_alpha = alphas[max_curv + 1]
    # plt.plot(curv)
    # plt.title('Max Double Fission LCurve 2nd Deriv @ alpha = ' + str(max_alpha))

    # recon_double = algorithms.solve_tikhonov(double_probs.reshape((-1)), response_double.reshape((response_double.shape[0], -1)).T, alpha=0.00240)
    # recon_double = recon_double.reshape(grid.num_y, grid.num_x)
    # plt.figure(figsize=(6, 4))
    # plt.imshow(recon_double, interpolation='none', extent=[-25./2, 25./2, -15./2, 15./2], cmap='viridis', vmin=0, vmax=1.0)
    # plt.title('Double Neutron Reconstruction')
    # cbar = plt.colorbar()
    # cbar.ax.get_yaxis().labelpad = 20
    # cbar.ax.set_ylabel(r'$\mu_{fission} / \mu_{total}$', rotation=270, size=17)
    # plt.tight_layout()

    # recon_simultaneous = algorithms.simultaneous_solve(single_probs.T, response_single.T,
    #                                                    double_probs.T, response_double.T, alpha=1)
    # recon_simultaneous = recon_simultaneous.reshape(grid.num_y, grid.num_x)
    # plt.figure()
    # plt.imshow(recon_simultaneous, interpolation='none', aspect='auto')
    # plt.colorbar()
    # plt.title('Simultaneous Reconstruction')

    # construct simultaneous reconstruction

    plt.show()
