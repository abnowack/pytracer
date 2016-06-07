import sys

import matplotlib.pyplot as plt
import numpy as np

from pytracer.transmission import *
from pytracer.grid import Grid
from geometries import build_shielded_geometry


def radon_scan_example(sim, n_angles):
    angles = np.linspace(0., 180., n_angles + 1)[:-1]

    r = radon(sim, angles)
    recon_image = inverse_radon(r, angles)

    extent = [-sim.detector.width / 2., sim.detector.width / 2.]

    plt.figure()
    plt.imshow(r, cmap=plt.cm.Greys_r, interpolation='none', aspect='auto')
    plt.xlabel('Angle')
    plt.ylabel('Radon Projection')
    plt.colorbar()

    plt.figure()

    plt.imshow(recon_image.T[::-1, :], interpolation='none', extent=extent * 2)
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.colorbar()

if __name__ == "__main__":
    sim = build_shielded_geometry()
    sim.grid = Grid(25, 15, 25, 15)
    sim.grid.create_mesh(225)
    sim.detector.width = 50
    sim.detector.nbins = 100
    angles = np.linspace(0., 180., 50 + 1)[:-1]

    # sim.rotate(20.)

    # plt.figure()
    # sim.draw()
    # plt.tight_layout()

    # plt.figure()
    # trace = sim.radon_transform(([11.01]))
    # plt.plot(trace[:, 0])

    # radon_scan_example(sim, 200)

    response = build_transmission_response(sim, angles)

    # plt.figure()
    # plt.imshow(response[:, :, 0], interpolation='none')

    measurement = scan(sim, angles)

    # plt.figure()
    # plt.imshow(measurement, interpolation='none')

    recon = recon_tikhonov(measurement, response)
    recon = recon.reshape(sim.grid.ny, sim.grid.nx)

    # plt.figure()
    # plt.imshow(recon, interpolation='none')
    #
    # plt.show()
