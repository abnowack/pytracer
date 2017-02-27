"""
Show the transmission response for a single pixel
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scripts.assemblies import shielded_assembly
import pytracer.geometry as geo
import pytracer.transmission as transmission


def curvature(x, y):
    xdot = x[:-1] - x[1:]
    ydot = y[:-1] - y[1:]
    xdotdot = xdot[:-1] - xdot[1:]
    ydotdot = ydot[:-1] - ydot[1:]
    xdot = (xdot[:-1] + xdot[1:]) / 2.
    ydot = (ydot[:-1] + ydot[1:]) / 2.

    num = np.abs(xdot * ydotdot - ydot * xdotdot)
    denom = np.power(xdot * xdot + ydot * ydot, 1.5)

    return np.divide(num, denom)

if __name__ == "__main__":
    assembly_solids = shielded_assembly()
    assembly_flat = geo.flatten(assembly_solids)

    plt.figure()
    geo.draw(assembly_solids)

    grid = geo.Grid(width=25, height=15, num_x=50, num_y=30)
    grid.draw()

    radians = np.linspace(0, np.pi, 100)
    arc_radians = np.linspace(-np.pi / 8, np.pi / 8, 100)
    source, detector_points, extent = geo.fan_beam_paths(60, arc_radians, radians, extent=True)

    # discretized transmission response
    response = transmission.grid_response(assembly_flat, grid, source, detector_points)

    cell_response = response[520 / 4 + 5]
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(cell_response, interpolation='none', extent=extent, cmap='viridis')
    plt.title('Pixel Radon Transform', size=20)
    plt.xlabel(r'Detector Angle $\theta$ (rad)', size=18)
    plt.ylabel(r'Neutron Angle $\phi$ (rad)', size=18)
    # cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(r'$A_{response}$', size=24, labelpad=15)

    plt.tight_layout()

    plt.show()
