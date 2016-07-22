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

    plt.figure()
    geo.draw(assembly_solids)

    radians = np.linspace(0, np.pi, 100)
    arc_radians = np.linspace(-np.pi / 8, np.pi / 8, 100)
    source, detector_points, extent = geo.fan_beam_paths(60, arc_radians, radians, extent=True)
    source = source[0, :, :]

    grid = geo.Grid(width=25, height=15, num_x=50, num_y=30)
    grid.draw()

    # i = 109
    # grid_points = grid.cell(i)
    # plt.scatter(grid_points[:, 0], grid_points[:, 1], zorder=12)
    # print(grid_points)
    #
    # unit_m = geo.Material('black', 1, 1)
    # vacuum = geo.Material('white', 0, 0)
    # cell_geom = [geo.Solid(geo.convert_points_to_segments(grid.cell(i)), unit_m, vacuum)]
    # cell_flat = geo.flatten(cell_geom)
    #
    # p = fission.grid_response_single_scan(source, detector_points, detector_points, cell_flat, assembly_flat, 0.2)

    # response = fission.grid_response_single(source, detector_points, detector_points, grid, assembly_flat, 0.2)
    # np.save('fission_response', response)
    response = np.load('fission_response.npy')

    # plt.figure()
    # plt.imshow(response[109].T, interpolation='none', extent=extent)

    # single_probs = fission.scan_single(source, detector_points, detector_points, assembly_flat, 0.2)
    # np.save('single_probs', single_probs)
    single_probs = np.load('single_probs.npy')

    plt.figure()
    plt.imshow(single_probs.T, interpolation='none', extent=extent)
    plt.colorbar()
    plt.title('Single Neutron Probability')
    plt.xlabel('Detector Orientation')
    plt.ylabel('Relative Neutron Angle')
    plt.tight_layout()

    recon = algorithms.solve_tikhonov(single_probs.T, response.T, alpha=1500)
    recon = recon.reshape(grid.num_y, grid.num_x)
    plt.figure()
    plt.imshow(recon, interpolation='none', aspect='auto')

    plt.show()
