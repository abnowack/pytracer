import sys
import matplotlib.pyplot as plt
import numpy as np
from scripts.assemblies import shielded_assembly
import pytracer.geometry as geo
import pytracer.transmission as transmission
import pytracer.algorithms as algorithms

if __name__ == "__main__":
    assembly_solids = shielded_assembly()
    assembly_flat = geo.flatten(assembly_solids)

    plt.figure()
    geo.draw(assembly_solids)

    grid = geo.Grid(width=25, height=15, num_x=50, num_y=30)
    grid.draw()

    radians = np.linspace(0, np.pi, 100)
    start, end, extent = geo.parallel_beam_paths(height=30, num_projections=200, offset=30, radians=radians,
                                                 extent=True)
    # for (s, e) in zip(start[:, 0], end[:, 0]):
    #     plt.plot([s[0], e[0]], [s[1], e[1]], color='blue')

    measurement = transmission.scan(assembly_flat, start, end)

    # discretized transmission response
    response = transmission.grid_response(assembly_flat, grid, start, end)
    plt.figure()
    plt.imshow(response[100, :, :], interpolation='none', extent=extent, aspect='auto')

    # reconstruction
    for alpha in [1, 10, 100, 1000]:
        recon = algorithms.solve_tikhonov(measurement.T, response.T, alpha=alpha)
        recon = recon.reshape(grid.num_y, grid.num_x)
        plt.figure()
        plt.imshow(recon, interpolation='none', extent=extent, aspect='auto')
        plt.title('alpha = ' + str(alpha))

    plt.show()
