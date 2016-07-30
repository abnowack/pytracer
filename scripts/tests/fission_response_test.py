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

    # grid = geo.Grid(width=25, height=15, num_x=50, num_y=30)
    grid = geo.Grid(width=25, height=15, num_x=25, num_y=15)
    grid.draw()

    cell_i = 109
    unit_m = geo.Material('black', 1, 1)
    vacuum = geo.Material('white', 0, 0)
    grid_points = grid.cell(cell_i)
    cell_geom = [geo.Solid(geo.convert_points_to_segments(grid_points, circular=True), unit_m, vacuum)]
    cell_flat = geo.flatten(cell_geom)
    plt.scatter(grid_points[:, 0], grid_points[:, 1], zorder=12)

    grid_response = fission.grid_response_scan(source, detector_points, detector_points, cell_flat, assembly_flat, 1,
                                               0.2)

    plt.figure()
    plt.imshow(grid_response.T, interpolation='none', extent=extent)
    plt.title('Grid Response')

    plt.show()
