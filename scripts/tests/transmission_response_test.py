import sys
import matplotlib.pyplot as plt
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

    # plt.figure()
    # geo.draw(assembly_solids)

    grid = geo.Grid(width=25, height=15, num_x=50, num_y=30)
    # grid.draw()

    radians = np.linspace(0, np.pi, 25)
    start, end, extent = geo.parallel_beam_paths(height=30, num_projections=200, offset=30, radians=radians,
                                                 extent=True)
    # for (s, e) in zip(start[:, 0], end[:, 0]):
    #     plt.plot([s[0], e[0]], [s[1], e[1]], color='blue')

    measurement = transmission.scan(assembly_flat, start, end)
    print('measurement ', measurement.sum())
    measurement_mean = np.mean(measurement)
    measurement_variance = np.std(measurement)
    scaled_measurement = (measurement - measurement_mean) / measurement_variance
    # plt.figure()
    # plt.imshow(scaled_measurement, interpolation='none', aspect='auto', extent=extent)

    # discretized transmission response
    response = transmission.grid_response(assembly_flat, grid, start, end)

    alphas = np.linspace(0, 2, 100)
    lcurve_y, lcurve_x = algorithms.trace_lcurve(measurement, response, alphas)
    plt.figure()
    plt.loglog(lcurve_x, lcurve_y)

    plt.figure()
    curv = algorithms.lcurve_curvature(np.log(lcurve_x), np.log(lcurve_y))
    max_curv = np.argmax(curv)
    max_alpha = alphas[max_curv + 1]
    plt.plot(curv)
    plt.title('Max LCurve 2nd Deriv @ alpha = ' + str(max_alpha))

    recon = algorithms.solve_tikhonov(measurement.T, response.T, alpha=max_alpha)
    recon = recon.reshape(grid.num_y, grid.num_x)
    plt.figure()
    plt.imshow(recon, interpolation='none', extent=extent, aspect='auto')
    plt.title('alpha = ' + str(max_alpha))
    plt.colorbar()

    plt.show()
