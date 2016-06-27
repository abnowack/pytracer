import sys
import matplotlib.pyplot as plt
import numpy as np
from pytracer.transmission import *
from pytracer.grid import Grid
from scripts.geometries import build_shielded_geometry

def radon_scan_example(sim, n_angles):
    r, angles = radon(sim, n_angles)
    recon_image = inverse_radon(r, angles)

    extent = [-sim.detector.width / 2, sim.detector.width / 2]

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
    sim.grid = Grid(25, 15, 35, 25)
    sim.detector.width = 25
    sim.detector.nbins = 200

    sim.draw()

    # radon_scan_example(sim, 200)

    response = build_transmission_response(sim, 100)
    plt.figure()
    plt.imshow(response[:, :, 0], interpolation='none')

    measurement, angles = radon(sim, 100)

    plt.figure()
    plt.imshow(measurement, interpolation='none')

    for alpha in [500, 1000, 1500, 2000, 2500]:
        recon = recon_tikhonov(measurement, response, alpha=alpha)
        recon = recon.reshape(sim.grid.ny, sim.grid.nx)
        plt.figure()
        plt.imshow(recon, interpolation='none')
        plt.title('alpha = ' + str(alpha))

    plt.show()
