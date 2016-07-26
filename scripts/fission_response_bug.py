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

    cell_i = 109
    grid_points = grid.cell(cell_i)
    plt.scatter(grid_points[:, 0], grid_points[:, 1], zorder=12)

    # response_single = np.load('fission_response_single.npy')
    #
    # plt.figure()
    # plt.imshow(response_single[cell_i].T, interpolation='none', extent=extent)
    # plt.title('Single Fission Response')
    #
    # arc_rad_i, rad_i = 31, 30

    unit_m = geo.Material('black', 1, 1)
    vacuum = geo.Material('white', 0, 0)
    cell_geom = [geo.Solid(geo.convert_points_to_segments(grid.cell(cell_i), True), unit_m, vacuum)]
    cell_flat = geo.flatten(cell_geom)
    # detector_segments = geo.convert_points_to_segments(detector_points[:, rad_i])
    # result = fission.grid_cell_response_single(source[rad_i], detector_points[arc_rad_i, rad_i], detector_segments,
    #                                            cell_flat, assembly_flat, 0.2)
    # print(result)

    plt.figure()
    geo.draw(assembly_solids)
    # plt.scatter(source[rad_i, 0], source[rad_i, 1])
    # plt.plot(detector_points[:, rad_i, 0], detector_points[:, rad_i, 1])
    # plt.plot([source[rad_i, 0], detector_points[arc_rad_i, rad_i, 0]],
    #          [source[rad_i, 1], detector_points[arc_rad_i, rad_i, 1]])
    plt.scatter(grid_points[:, 0], grid_points[:, 1], zorder=12)

    # segment, val = fission.find_fission_segments(source[rad_i], detector_points[arc_rad_i, rad_i], cell_flat)
    # segment = segment[0]
    # plt.plot([segment[0, 0], segment[1, 0]], [segment[0, 1], segment[1, 1]], color='black')
    # plt.scatter([segment[0, 0], segment[1, 0]], [segment[0, 1], segment[1, 1]], color='black')
    #
    # intersects, indexes = transmission.intersections(source[rad_i], detector_points[arc_rad_i, rad_i],
    #                                                  cell_flat.segments)
    # plt.scatter(intersects[:, 0], intersects[:, 1], color='red')
    geo.draw(cell_geom, True)

    print(grid.cell(cell_i))
    print(cell_flat.segments)
    # print(segment)

    plt.show()
