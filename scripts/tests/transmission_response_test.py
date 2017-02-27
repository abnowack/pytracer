import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scripts.assemblies import shielded_assembly
import pytracer.geometry as geo
import pytracer.transmission as transmission
import pytracer.algorithms as algorithms


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

    # for (s, e) in zip(start[:, 0], end[:, 0]):
    #     plt.plot([s[0], e[0]], [s[1], e[1]], color='blue')

    measurement = transmission.scan(assembly_flat, source, detector_points)
    # print('measurement ', measurement.sum())
    # measurement_mean = np.mean(measurement)
    # measurement_variance = np.std(measurement)
    # scaled_measurement = (measurement - measurement_mean) / measurement_variance
    plt.figure()
    plt.imshow(measurement, interpolation='none', aspect='auto', extent=extent)

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

    # alphas = np.linspace(0, 2, 100)
    # lcurve_y, lcurve_x = algorithms.trace_lcurve(measurement, response, alphas)
    # plt.figure()
    # plt.loglog(lcurve_x, lcurve_y)

    # plt.figure()
    # curv = algorithms.lcurve_curvature(np.log(lcurve_x), np.log(lcurve_y))
    # max_curv = np.argmax(curv)
    # max_alpha = alphas[max_curv + 1]
    # plt.plot(curv)
    # plt.title('Max LCurve 2nd Deriv @ alpha = ' + str(max_alpha))

    # d1 = measurement.reshape((-1))
    # G1 = response.reshape((response.shape[0], -1)).T
    # recon = algorithms.solve_tikhonov(d1, G1, alpha=0.4)
    # recon = recon.reshape(grid.num_y, grid.num_x)
    # plt.figure(figsize=(6, 4))
    # plt.imshow(recon, interpolation='none', extent=[-25. / 2, 25. / 2, -15. / 2, 15. / 2], cmap='viridis')
    # plt.title('Transmission Reconstruction')
    # cb = plt.colorbar()
    # cb.set_label(r'Macro cross section $\mu_{total}$', size=15, labelpad=15)
    # plt.xlabel('X (cm)')
    # plt.ylabel('Y (cm)')

    plt.tight_layout()

    plt.show()
