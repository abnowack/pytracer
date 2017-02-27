"""
Calculate fission responses for a single grid cell and display the results
"""

import sys
import matplotlib.pyplot as plt
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

    ax1.imshow(data1, interpolation='none', extent=extent, cmap='viridis')
    ax1.set_title(title1)

    ax2.imshow(data2, interpolation='none', extent=extent, cmap='viridis')
    ax2.set_title(title2)

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

    avg_nu = np.array([0.0481677, 0.2485215, 0.4253044, 0.2284094, 0.0423438, 0.0072533], dtype=np.double)
    avg_nu /= np.sum(avg_nu)

    plt.figure()
    geo.draw(assembly_solids)

    radians = np.linspace(0, np.pi, 200)
    arc_radians = np.linspace(-np.pi / 8, np.pi / 8, 200)
    source, detector_points, extent = geo.fan_beam_paths(60, arc_radians, radians, extent=True)
    source = source[0, :, :]

    # grid = geo.Grid(width=25, height=15, num_x=50, num_y=30)
    grid = geo.Grid(width=25, height=15, num_x=50, num_y=30)
    grid.draw()

    cell_i = 109 * 4 - 20
    unit_m = geo.Material('black', 1, 1)
    vacuum = geo.Material('white', 0, 0)
    grid_points = grid.cell(cell_i)
    cell_geom = [geo.Solid(geo.convert_points_to_segments(grid_points, circular=True), unit_m, vacuum)]
    cell_flat = geo.flatten(cell_geom)
    plt.scatter(grid_points[:, 0], grid_points[:, 1], zorder=12)

    single_response = fission.grid_response_scan(source, detector_points, detector_points, cell_flat, assembly_flat, 1,
                                                 avg_nu)
    double_response = fission.grid_response_scan(source, detector_points, detector_points, cell_flat, assembly_flat, 2,
                                                 avg_nu)

    nice_double_plot(single_response.T, double_response.T, extent, 'Single Fission Neutron Response',
                     'Double Fission Neutron Response',
                     'Detector Orientation Angle', 'Source Neutron Direction Angle')

    plt.show()