import sys
import matplotlib.pyplot as plt
import numpy as np
from scripts.assemblies import shielded_assembly
import pytracer.geometry as geo
import pytracer.transmission as transmission
import pytracer.algorithms as algorithms
import pytracer.fission as fission

if __name__ == "__main__":
    assembly_solids = shielded_assembly()
    assembly_flat = geo.flatten(assembly_solids)

    # plt.figure()
    # geo.draw(assembly_solids)

    radians = np.linspace(0, np.pi, 100)
    arc_radians = np.linspace(-np.pi / 8, np.pi / 8, 100)
    source, detector_points, extent = geo.fan_beam_paths(60, arc_radians, radians, extent=True)
    source = source[0, :, :]

    # grid = geo.Grid(width=25, height=15, num_x=50, num_y=30)
    grid = geo.Grid(width=25, height=15, num_x=25, num_y=15)
    # grid.draw()

    cell_i = 109
    grid_points = grid.cell(cell_i)
    # plt.fill(grid_points[:, 0], grid_points[:, 1], color='red', alpha=0.5, zorder=12)

    # response_single = fission.grid_response(source, detector_points, detector_points, grid, assembly_flat, 1, 0.2)
    # np.save(r'data\fission_response_single', response_single)
    response_single = np.load(r'data\fission_response_single.npy')

    # plt.figure()
    # plt.imshow(response_single[cell_i].T, interpolation='none', extent=extent)
    # plt.title('Single Fission Response')

    # response_double = fission.grid_response(source, detector_points, detector_points, grid, assembly_flat, 2, 0.2)
    # np.save(r'data\fission_response_double', response_double)
    response_double = np.load(r'data\fission_response_double.npy')

    # plt.figure()
    # plt.imshow(response_double[cell_i].T, interpolation='none', extent=extent)
    # plt.title('Double Fission Response')

    # single_probs = fission.scan(source, detector_points, detector_points, assembly_flat, 1, 0.2)
    # np.save(r'data\single_probs', single_probs)
    single_probs = np.load(r'data\single_probs.npy')

    # double_probs = fission.scan(source, detector_points, detector_points, assembly_flat, 2, 0.2)
    # np.save(r'data\double_probs', double_probs)
    double_probs = np.load(r'data\double_probs.npy')

    # plt.figure()
    # plt.imshow(single_probs.T, interpolation='none', extent=extent)
    # plt.colorbar()
    # plt.title('Single Neutron Probability')
    # plt.xlabel('Detector Orientation')
    # plt.ylabel('Relative Neutron Angle')
    # plt.tight_layout()
    #
    # plt.figure()
    # plt.imshow(double_probs.T, interpolation='none', extent=extent)
    # plt.colorbar()
    # plt.title('Double Neutron Probability')
    # plt.xlabel('Detector Orientation')
    # plt.ylabel('Relative Neutron Angle')
    # plt.tight_layout()

    # alpha = 0.001
    #

    # alphas = np.linspace(1e-7, 100, 500)
    # norm, residual = algorithms.trace_lcurve(single_probs, response_single, alphas)
    # plt.figure()
    # plt.loglog(residual, norm, marker='o')
    # plt.title('LCurve Single Fission')
    #
    # plt.figure()
    # curv = algorithms.lcurve_curvature(np.log(residual), np.log(norm))
    # max_curv = np.argmax(curv)
    # max_alpha = alphas[max_curv + 1]
    # plt.plot(curv)
    # plt.title('Max Single Fission LCurve 2nd Deriv @ alpha = ' + str(max_alpha))
    #
    # recon_single = algorithms.solve_tikhonov_direct(single_probs.T, response_single.T, alpha=1)
    # recon_single = recon_single.reshape(grid.num_y, grid.num_x)
    # plt.figure()
    # plt.imshow(recon_single, interpolation='none', aspect='auto')
    # plt.colorbar()
    # plt.title('Single Reconstruction')
    #
    # alphas = np.linspace(1e-7, 100, 500)
    # norm, residual = algorithms.trace_lcurve(double_probs, response_double, alphas)
    # plt.figure()
    # plt.loglog(residual, norm, marker='o')
    # plt.title('LCurve Double Fission')
    #
    # plt.figure()
    # curv = algorithms.lcurve_curvature(np.log(residual), np.log(norm))
    # max_curv = np.argmax(curv)
    # max_alpha = alphas[max_curv + 1]
    # plt.plot(curv)
    # plt.title('Max Double Fission LCurve 2nd Deriv @ alpha = ' + str(max_alpha))
    #
    # recon_double = algorithms.solve_tikhonov_direct(double_probs.T, response_double.T, alpha=1)
    # recon_double = recon_double.reshape(grid.num_y, grid.num_x)
    # plt.figure()
    # plt.imshow(recon_double, interpolation='none', aspect='auto')
    # plt.colorbar()
    # plt.title('Double Reconstruction')

    recon_simultaneous = algorithms.simultaneous_solve(single_probs.T, response_single.T,
                                                       double_probs.T, response_double.T, alpha=1)
    recon_simultaneous = recon_simultaneous.reshape(grid.num_y, grid.num_x)
    plt.figure()
    plt.imshow(recon_simultaneous, interpolation='none', aspect='auto')
    plt.colorbar()
    plt.title('Simultaneous Reconstruction')

    plt.show()
